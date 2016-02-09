import argparse
import itertools
import os

def frange(start, stop, step):
    # extension of range to floats
    curr = start
    while curr <= stop:
        yield curr
        curr += step

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', help='Experiment output file')
    subparsers = parser.add_subparsers()

    nb_parser = subparsers.add_parser('nb')
    nb_parser.add_argument('--alpha_start', type=float, help='Start value for alpha')
    nb_parser.add_argument('--alpha_step', type=float, help='Step value for alpha')
    nb_parser.add_argument('--alpha_end', type=float, help='End value for alpha')
    nb_parser.set_defaults(classifier='nb', alpha_start=0, alpha_step=1, alpha_end=1)

    logreg_parser = subparsers.add_parser('logreg')
    logreg_parser.add_argument('--params', help='Parameters to grid search on', nargs='*')
    logreg_parser.add_argument('--param_starts', type=float, help='Parameter start values', nargs='*')
    logreg_parser.add_argument('--param_steps', type=float, help='Parameter step values', nargs='*')
    logreg_parser.add_argument('--param_ends', type=float, help='Parameter end values', nargs='*')
    logreg_parser.add_argument('--eta', type=float, help='Static learning rate value')
    logreg_parser.add_argument('--batch_size', type=float, help='Static minibatch size value')
    logreg_parser.add_argument('--max_epochs', type=float, help='Static max epochs value')
    logreg_parser.add_argument('--l', type=float, help='Static regularization value')
    logreg_parser.set_defaults(classifier='logreg', eta=0.01, batch_size=100, max_epochs=50, l=1)

    hinge_parser = subparsers.add_parser('hinge')
    hinge_parser.add_argument('--params', help='Parameters to grid search on', nargs='*')
    hinge_parser.add_argument('--param_starts', type=float, help='Parameter start values', nargs='*')
    hinge_parser.add_argument('--param_steps', type=float, help='Parameter step values', nargs='*')
    hinge_parser.add_argument('--param_ends', type=float, help='Parameter end values', nargs='*')
    hinge_parser.add_argument('--eta', type=float, help='Static learning rate value')
    hinge_parser.add_argument('--batch_size', type=float, help='Static minibatch size value')
    hinge_parser.add_argument('--max_epochs', type=float, help='Static max epochs value')
    hinge_parser.add_argument('--l', type=float, help='Static regularization value')
    hinge_parser.set_defaults(classifier='hinge', eta=0.01, batch_size=100, max_epochs=50, l=1)

    args = vars(parser.parse_args())
    print(args)

    if args['classifier'] == 'nb':
        grid = [i for i in frange(args['alpha_start'], args['alpha_end'], args['alpha_step'])]
        for alpha in grid:
            print(alpha)
            divider = '\"===========\"'
            command = 'luajit HW1.lua -datafile SST1.hdf5 -classifier nb -alpha %s >> %s' % (alpha, args['output'])
            os.system(command)
            divide_command = 'echo %s >> %s' % (divider, args['output'])
            os.system(divide_command)
    else:
        # take care of naming issue
        args['lambda'] = args['l']
        del args['l']

        axes = []
        for i in range(len(args['params'])):
            if args['params'][i] == 'eta':
                axes.append([10**i for i in frange(args['param_starts'][i], args['param_ends'][i], args['param_steps'][i])])
            else:
                axes.append([i for i in frange(args['param_starts'][i], args['param_ends'][i], args['param_steps'][i])])
        grid = [list(i) for i in itertools.product(*axes)]
        for point in grid:
            print(point)
            divider = '\"===========\"'
            command = 'luajit HW1.lua -datafile SST1.hdf5 -classifier %s' % args['classifier']
            for i in range(len(args['params'])):
                param_name = args['params'][i]
                param_val = str(point[i])
                command_snippet = ' -' + param_name + ' ' + param_val
                command += command_snippet
            command += ' >> ' + args['output']
            print(command)
            os.system(command)
            divide_command = 'echo %s >> %s' % (divider, args['output'])
            os.system(divide_command)
