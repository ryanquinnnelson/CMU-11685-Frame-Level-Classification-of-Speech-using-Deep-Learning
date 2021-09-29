class PhaseHandler:

    def __init__(self, load_from_checkpoint, first_epoch, num_epochs, trainer, evaluator, tester, checkpoint_file=None):
        self.load_from_checkpoint = load_from_checkpoint
        self.checkpoint_file = checkpoint_file
        self.first_epoch = first_epoch
        self.num_epochs = num_epochs


    def _load_checkpoint(self, model, optimizer, scheduler, statshandler, checkpointhandler, devicehandler):
        device = devicehandler.get_device()
        checkpoint = checkpointhandler.load(self.checkpoint_file, device, model, optimizer, scheduler)

        # restore stats
        statshandler.stats = checkpoint['stats']

        # set which epoch to start from
        self.first_epoch = checkpoint['next_epoch']

    def process_epochs(self, train_loader, val_loader, test_loader, model, optimizer, scheduler, loss_func,
                       evaluate_batch_func, evaluate_epoch_func, format_func,
                       datahandler, devicehandler, checkpointhandler, schedulerhandler, wandbconnector):

        # load checkpoint if necessary
        if self.load_from_checkpoint:
            self._load_checkpoint(devicehandler, checkpointhandler, model, optimizer, scheduler)

        # run epochs
        for epoch in range(self.first_epoch, self.num_epochs + 1):
            # record start time
            start = time.time()

            # train
            train_loss = train_model(epoch, self.num_epochs, train_loader, model, loss_func, devicehandler,
                                     optimizer)

            # validate
            val_loss, val_acc = evaluate_model(epoch, self.num_epochs, val_loader, model, loss_func,
                                               evaluate_batch_func, evaluate_epoch_func, devicehandler)

            # test
            out = test_model(epoch, self.num_epochs, test_loader, model, devicehandler)
            out = format_func(out)
            datahandler.save(out, epoch)

            # stats
            end = time.time()
            self._collect_stats(epoch, train_loss, val_loss, val_acc, start, end)
            self._report_stats(wandbconnector)

            # scheduler
            schedulerhandler.update_scheduler(scheduler, val_acc)

            # save model checkpoint
            checkpointhandler.save(model, optimizer, scheduler, epoch + 1, self.stats)

            # check if early stopping criteria is met
            if self._stopping_criteria_is_met(epoch, wandbconnector):
                logging.info('Early stopping criteria is met. Stopping the training process...')
                break  # stop running epochs
