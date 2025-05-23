from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook
from mmrotate.utils import get_root_logger

@HOOKS.register_module()
class PrintThres(Hook):
    def after_train_iter(self, runner):
        if is_module_wrapper(runner.model):
            model = runner.model.module
        else:
            model = runner.model
        iter = model.iter_count
        if iter < 13301 or (iter>=40000 and iter<40500) or (iter>=80000 and iter<80500):
            score = model.semi_loss.mean_score
            if score is not None:
                logger = get_root_logger()
                if iter < 13301:
                   logger.info(f"mean_score_early: {score}")
                elif iter>=40000 and iter<40500:
                   logger.info(f"mean_score_middle: {score}")
                else:
                   logger.info(f"mean_score_late: {score}")

            else:
                pass
        else:
            pass