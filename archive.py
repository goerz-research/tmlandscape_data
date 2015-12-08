#!/usr/bin/env python
"""Create tgz archives for the given folders, possibly limiting data to that of
only one or more stages.
"""
import os
import sys
import click


ALLOWED_STAGE_VALUES = ['1', '2', '3', '4', '4_1freq', 'prop']


def validate_stage(ctx, param, values):
    stage_names = []
    for value in values:
        if value not in ALLOWED_STAGE_VALUES:
            raise click.BadParameter('Allowed values are: %s'
                                     % str(ALLOWED_STAGE_VALUES))
        if value =='prop':
            stage_names.append('stage_prop')
        else:
            stage_names.append('stage'+value)
    return stage_names


@click.command(help=__doc__)
@click.help_option('--help', '-h')
@click.option('--stage', help="Include data for the given stage in the "
    "output. May be given multiple times. If not given at all, include "
    "all data. Allowed values are %s"%str(ALLOWED_STAGE_VALUES),
    metavar='STAGE', multiple=True, callback=validate_stage)
@click.option('--dry-run', '-n', is_flag=True, help="Perform a dry run")
@click.argument('folders', nargs=-1, type=click.Path(exists=True))
def archive(folders, stage, dry_run):
    if len(stage) == 0:
        exclude = set([])
    else:
        exclude = set(['stage1', 'stage2', 'stage3', 'stage4', 'stage4_1freq',
                       'stage_prop'])
        for stage_name in stage:
            exclude.remove(stage_name)

    for folder in folders:

        outfile = folder
        for s in stage:
            outfile += '_%s' % s
        outfile += '.tgz'

        cmd = ('tar '+' '.join("--exclude=%s"%s for s in list(exclude))
               + " -c " + folder + " | gzip > " + outfile)

        click.echo(cmd)
        if not dry_run:
            os.system(cmd)


if __name__ == "__main__":
    sys.exit(archive())
