import logging


class StatsHandler:

    def __init__(self, val_metric_name, comparison_metric, comparison_best_is_max, comparison_patience):
        self.stats = {'epoch': [],
                      'runtime': [],
                      'train_loss': [],
                      'val_loss': [],
                      val_metric_name: []}
        self.val_metric_name = val_metric_name
        self.comparison_metric = comparison_metric
        self.best_is_max = comparison_best_is_max
        self.comparison_patience = comparison_patience
        self.num_epochs_worse_than_best_model = 0

    def _model_is_worse_by_comparison_metric(self, epoch, wandbconnector):

        # check whether the metric we are using for comparison against other runs
        # is better than other runs for this epoch
        logging.info('Checking whether comparison metric for current model is worse than the best model so far...')
        best_name, best_val = wandbconnector.get_best_value(self.comparison_metric, epoch, self.best_is_max)
        model_name = wandbconnector.run_name
        model_val = self.stats[self.comparison_metric][-1]
        print(f'best:\t{best_name}\t{best_val}\nmodel:\t{model_name}\t{model_val}')

        if best_val is not None and self.best_is_max:
            # compare values for this epoch
            if model_val < best_val:
                model_is_worse = True
                logging.info('A previous model has the best value for the comparison metric for this epoch.')
            else:
                logging.info('Current model has the best value for the comparison metric for this epoch.')
                model_is_worse = False
        elif best_val is not None and not self.best_is_max:  # best is min
            # compare values for this epoch
            if model_val > best_val:
                model_is_worse = True
                logging.info('A previous model has the best value for the comparison metric for this epoch.')
            else:
                logging.info('Current model has the best value for the comparison metric for this epoch.')
                model_is_worse = False
        else:
            # no model to compare against for this epoch
            model_is_worse = False

        return model_is_worse

    def stopping_criteria_is_met(self, epoch, wandbconnector):
        logging.info('Checking early stopping criteria...')

        # criteria 1 - metric comparison
        model_is_worse = self._model_is_worse_by_comparison_metric(epoch, wandbconnector)
        if model_is_worse:
            self.num_epochs_worse_than_best_model += 1
            logging.info('Number of epochs in a row this model is worse than best model' +
                         f':{self.num_epochs_worse_than_best_model}')
            logging.info('Number of epochs in a row this model can be worse than best model ' +
                         f'before stopping:{self.comparison_patience}')
        else:
            self.num_epochs_worse_than_best_model = 0  # reset value

        # check all criteria to determine if we need to stop learning
        if self.num_epochs_worse_than_best_model > self.comparison_patience:
            stopping_criteria_met = True
            logging.info('Early stopping criteria is met.')
        else:
            stopping_criteria_met = False
            logging.info('Early stopping criteria is not met.')

        return stopping_criteria_met

    def report_stats(self, wandbconnector):

        # save epoch stats to wandb
        epoch_stats_dict = dict()
        for key in self.stats.keys():
            epoch_stats_dict[key] = self.stats[key][-1]  # latest value
        wandbconnector.log_stats(epoch_stats_dict)
        logging.info(f'stats:{epoch_stats_dict}')

    def collect_stats(self, epoch, train_loss, val_loss, val_metric, start, end):

        # calculate runtime
        runtime = end - start

        # update stats
        self.stats['epoch'].append(epoch)
        self.stats['runtime'].append(runtime)
        self.stats['train_loss'].append(train_loss)
        self.stats['val_loss'].append(val_loss)
        self.stats[self.val_metric_name].append(val_metric)
