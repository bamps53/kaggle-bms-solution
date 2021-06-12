

def lrfn(step, warmup_lr, init_lr, final_lr, num_total_steps, warmup_steps):
    if step < warmup_steps:
        warmup_factor = (step / warmup_steps) ** 2
        lr = warmup_lr + (init_lr - warmup_lr) * warmup_factor
    else:
        power = (step - warmup_steps) // ((num_total_steps -
                                           warmup_steps) / (num_total_steps + 1))
        decay_factor = ((init_lr / final_lr) ** (1 / num_total_steps)) ** power
        lr = init_lr / decay_factor

    return round(lr, 8)


class LRReduce():
    def __init__(self, optimizer, lr_schedule):
        self.opt = optimizer
        self.lr_schedule = lr_schedule
        # assign initial learning rate
        self.lr = lr_schedule[0]
        self.opt.learning_rate.assign(self.lr)

    def step(self, step, loss=None):
        self.lr = self.lr_schedule[step]
        # assign learning rate to optimizer
        self.opt.learning_rate.assign(self.lr)

    def get_counter(self):
        return self.c

    def get_lr(self):
        return self.lr


def get_scheduler(optimizer, warmup_lr, init_lr, final_lr, num_total_steps, warmup_steps):
    lr_fn = [lrfn(step, warmup_lr, init_lr, final_lr, num_total_steps, warmup_steps)
             for step in range(num_total_steps)]
    scheduler = LRReduce(optimizer, lr_fn)
    return scheduler
