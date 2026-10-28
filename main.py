from pyfedappwrap.engine.runtime import FedDBEngine

from app import LogisticRegressionAPP

engine = FedDBEngine()
engine.register(LogisticRegressionAPP())

if __name__ == '__main__':
    engine.start()
    engine.wait_until_stop()
