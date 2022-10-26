"""Implements Adapter Controller, a module that keeps multiple
layers of Adapters, and controls which adapter layer to use."""
import os
import torch.nn as nn
from .adapter_modeling import (Adapter, HyperComplexAdapter, LowRankAdapter, 
                                KernelAdapter, LoRA, LoraAdapter, PrefixAdapter)


class AdapterController(nn.Module):
    """Implements Adapter controller module which controls the logics of
    putting adapter layers within transformer's layers."""

    def __init__(self, config):
        # config here is adapter_config (json file)
        super().__init__()

        self.config = config
        self.adapters = nn.ModuleDict(dict())
        self.tasks = config.tasks
        self.device = config.device
        
        self.parallel_adapter = config.parallel_adapter

        self.shared_phm_rule = config.shared_phm_rule
        self.hypercomplex_adapters = config.hypercomplex_adapters

        self.low_rank_adapters = config.low_rank_adapters # abandoned, for old lora
        # self.prefix_tuning = config.prefix_tuning

        self.kernel_adapter = config.kernel_adapter
        
        self.lora = (config.lora != "")
        
        # for the original prefix-tuning ("reparametrization"), adapter_controller won't be used
        # adapter controller would only be used for the new one ("landmark") or the ablation ("MH_adapter")
        self.prefix_tuning = (config.prefix_tuning == "landmark" or config.prefix_tuning == "MH_adapter"
                            or config.prefix_tuning == "value_gating")
        
        self.adapters = self.construct_adapters(self.tasks)
        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter
        if self.add_layer_norm_before_adapter and not self.kernel_adapter:
            self.pre_layer_norm = nn.LayerNorm(config.input_dim)
        if self.add_layer_norm_after_adapter and not self.kernel_adapter:
            self.post_layer_norm = nn.LayerNorm(config.input_dim)

        
        



    def get_task(self, task):
        return task 

    def construct_adapters(self, tasks):
        """
        Constructs adapter layers and adds them to a dictionary for the given
        tasks.
        Args:
            tasks: A list of string containing the task names.
        """
        for task in tasks:
            # the kernel adapters are exclusive to the others
            if self.kernel_adapter:
                self.adapters[task] = KernelAdapter(self.config)
            elif self.low_rank_adapters:
                self.adapters[task] = LowRankAdapter(self.config)
            elif self.hypercomplex_adapters:
                self.adapters[task] = HyperComplexAdapter(self.config)
            elif self.lora:
                self.adapters[task] = LoRA(self.config)
            elif self.prefix_tuning:
                self.adapters[task] = PrefixAdapter(self.config)
            else:
                self.adapters[task] = Adapter(self.config)
        return self.adapters

    def disable_adapters(self, tasks):
        """
        Given a list of tasks, it freezes their corresponding adapter layers'
        parameters.
        Args:
           tasks: List of tasks.
        """
        tasks = self.convert_to_list(tasks)
        for task in tasks:
            adapter = self.get_adapter(task)
            for param in adapter.parameters():
                param.requires_grad = False

    def convert_to_list(self, tasks):
        if isinstance(tasks, list):
            return tasks
        return [tasks]

    def enable_adapters(self, tasks):
        """
        Given a list of tasks, it unfreezes their corresponding adapter layers.
        Args:
            tasks: Given list of tasks.
        """

        tasks = self.convert_to_list(tasks)
        for task in tasks:
            adapter = self.get_adapter(task)
            for name, param in adapter.named_parameters():
                if self.config.hypercomplex_adapters and not self.config.learn_phm:
                    if not "phm_rule" in name:
                        param.requires_grad = True
                else:
                    param.requires_grad = True

    def get_adapter(self, task):
        """Given a task returns its corresponding adapter layer.
        Args:
            task: Input task name.
        Returns:
            Adapter layer corresponding to the given task.
        """
        return self.adapters[task]

    def forward(self, inputs, task):
        """
        Retrieves the adapter layer corresponding to the given
        task. It freezes the adapter layers for all the other tasks
        and call the selected adapter layer.
        Args:
            task: the name of the current task.
            inputs: the inputs to feed in in the adapter layer.
        Returns:
            outputs of the adapter layer.
        """
        task = self.get_task(task)
        # Enables the adapter layer for the given task.
        self.enable_adapters(task)
        # Disable other adapters.
        other_tasks = [x for x in self.tasks if x != task]
        self.disable_adapters(other_tasks)
        adapter = self.get_adapter(task)


        if self.kernel_adapter or self.lora or self.prefix_tuning:
            # Specialized forward call for kernel_adapters
            outputs = adapter(inputs)
            
        else:
            # General forms of bottleneck structure adapter forward calls
            z = self.pre_layer_norm(inputs) if self.add_layer_norm_before_adapter else inputs
            outputs = adapter(z)
            
            if self.parallel_adapter == "":     
                # change the original order in VLM
                outputs = outputs + inputs
            
            if self.add_layer_norm_after_adapter:
                outputs = self.post_layer_norm(outputs)
            # outputs = outputs + inputs
        return outputs





