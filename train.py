import time
import torch
from model.setting import Setting, Arguments
from model.mlm_model.processor import Processor


def main(args, logger) -> None:

    processor = Processor(args)
    config = processor.model_setting()
    logger.info('Model Setting Complete')

    if args.MLM == 'True':
        logger.info('==Start Masked Language Modeling==')

        loss, acc = processor.mlm()

        print(f'\n\t==MLM loss: {loss:.4f} | MLM acc: {acc:.4f} |==\n')

    else:
        if args.train == 'True':
            logger.info('Start Training')

            for epoch in range(args.epochs):
                start_time = time.time()

                train_loss, train_acc = processor.train()
                valid_loss, valid_acc = processor.valid()

                end_time = time.time()
                epoch_mins, epoch_secs = processor.metric.cal_time(start_time, end_time)

                performance = {'tl': train_loss, 'vl': valid_loss,
                               'tma': train_acc, 'vma': valid_acc,
                               'ep': epoch, 'epm': epoch_mins, 'eps': epoch_secs}

                processor.metric.save_model(config, performance, processor.model_checker)

                if processor.model_checker['early_stop']:
                    logger.info('Early Stopping')
                    break

        if args.test == 'True':
            logger.info("Start Test")

            test_loss, test_mem_acc = processor.test()
            print(f'\n\t==Test loss: {test_loss:.4f} | Test Classification acc: {test_mem_acc:.4f} |==\n')

            processor.metric.print_size_of_model(config['model'])
            processor.metric.count_parameters(config['model'])


if __name__ == '__main__':
    args, logger = Setting().run()
    main(args, logger)
