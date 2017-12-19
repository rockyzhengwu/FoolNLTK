#!/usr/bin/env python
#-*-coding:utf-8-*-


"""Fool command line interface."""
import sys
import fool
from argparse import ArgumentParser

parser = ArgumentParser(usage="%s -m fool [options] filename" % sys.executable, description="Fool command line interface.", epilog="If no filename specified, use STDIN instead.")
parser.add_argument("-d", "--delimiter", metavar="DELIM", default=' / ',
                    nargs='?', const=' ',
                    help="use DELIM instead of ' / ' for word delimiter; or a space if it is used without DELIM")
parser.add_argument("-p", "--pos", metavar="DELIM", nargs='?', const='_',
                    help="enable POS tagging; if DELIM is specified, use DELIM instead of '_' for POS delimiter")
parser.add_argument("-D", "--dict", help="use DICT as dictionary")
parser.add_argument("-u", "--user-dict",
                    help="use USER_DICT together with the default dictionary or DICT (if specified)")

parser.add_argument("filename", nargs='?', help="input file")

args = parser.parse_args()

delim = args.delimiter

fp = open(args.filename, 'r') if args.filename else sys.stdin
ln = fp.readline()

while ln:
    l = ln.rstrip('\r\n')
    result = delim.join(fool.cut(ln.rstrip('\r\n')))
    print(result)
    ln = fp.readline()

fp.close()
