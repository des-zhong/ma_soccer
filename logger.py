import sys


class Logger(object):
    def __init__(self, logFile="Default.log"):
        self.terminal = sys.stdout
        self.log = open(logFile, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
