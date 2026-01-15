import torch

def set_up_conv_optimizer(optim_settings, conv_parameters, linear_parameters):
    if optim_settings["mult_optimizers"]:
        # Configure linear optimizer (Muon)
        linear_kwargs = optim_settings["linear_optimizer"]["kwargs"].copy()

        # Fallback to torch.optim optimizers if Muon is not available
        linear_optimizer_class = getattr(
            torch.optim, optim_settings["linear_optimizer"]["name"]
        )
        linear_parameters = [{"params": linear_parameters}]
        linear_optimizer = linear_optimizer_class(
            linear_parameters, **linear_kwargs
        )

        # Configure conv optimizer
        conv_kwargs = optim_settings["conv_optimizer"]["kwargs"].copy()

        conv_optimizer_class = getattr(
            torch.optim, optim_settings["conv_optimizer"]["name"]
        )
        conv_parameters = [{"params": conv_parameters}]
        conv_optimizer = conv_optimizer_class(conv_parameters, **conv_kwargs)

        # Configure LR schedulers for manual optimization
        # For manual optimization, schedulers should be configured in the return dict
        linear_scheduler_config = None
        conv_scheduler_config = None

        linear_lr_scheduler = optim_settings["linear_optimizer"].get("lr_scheduler")
        if linear_lr_scheduler is not None:
            # Scheduler config dict
            scheduler_class = linear_lr_scheduler.get("class")
            scheduler_kwargs = linear_lr_scheduler.get("kwargs", {})
            if scheduler_class is not None:
                linear_scheduler = scheduler_class(
                    linear_optimizer, **scheduler_kwargs
                )
                linear_scheduler_config = {
                    "scheduler": linear_scheduler,
                    "interval": scheduler_kwargs.get("interval", "step"),
                    "frequency": scheduler_kwargs.get("frequency", 1),
                }

        conv_lr_scheduler = optim_settings["conv_optimizer"].get("lr_scheduler")
        if conv_lr_scheduler is not None:
            # Scheduler config dict
            scheduler_class = conv_lr_scheduler.get("class")
            scheduler_kwargs = conv_lr_scheduler.get("kwargs", {})
            if scheduler_class is not None:
                conv_scheduler = scheduler_class(conv_optimizer, **scheduler_kwargs)
                conv_scheduler_config = {
                    "scheduler": conv_scheduler,
                    "interval": scheduler_kwargs.get("interval", "step"),
                    "frequency": scheduler_kwargs.get("frequency", 1),
                }

        # Return optimizers with optional schedulers for manual optimization
        result = [{"optimizer": linear_optimizer}]
        if linear_scheduler_config is not None:
            result[0]["lr_scheduler"] = linear_scheduler_config

        result.append({"optimizer": conv_optimizer})
        if conv_scheduler_config is not None:
            result[1]["lr_scheduler"] = conv_scheduler_config

        return result
    else:
        # Single optimizer configuration
        optimizer_kwargs = optim_settings["optimizer"]["kwargs"].copy()

        optimizer_class = getattr(torch.optim, optim_settings["optimizer"]["name"])
        parameters = [
            {
                "params": list(linear_parameters)
                + list(conv_parameters)
            }
        ]
        optimizer = optimizer_class(parameters, **optimizer_kwargs)

        # Configure LR scheduler for single optimizer
        scheduler_config = None
        lr_scheduler = optim_settings["optimizer"].get("lr_scheduler")
        if lr_scheduler is not None:
            scheduler_class = lr_scheduler.get("class")
            scheduler_kwargs = lr_scheduler.get("kwargs", {})
            if scheduler_class is not None:
                scheduler = scheduler_class(optimizer, **scheduler_kwargs)
                scheduler_config = {
                    "scheduler": scheduler,
                    "interval": scheduler_kwargs.get("interval", "step"),
                    "frequency": scheduler_kwargs.get("frequency", 1),
                }

        result = [{"optimizer": optimizer}]
        if scheduler_config is not None:
            result[0]["lr_scheduler"] = scheduler_config

        return result