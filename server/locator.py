import logging
import os
import sys

import tornado
import tornado.ioloop
import tornado.web


DEBUG_MODE = False

if __name__ == "__main__":
    assert __file__
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    DEBUG_MODE = True
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s\t%(message)s', datefmt='%H:%M:%S')
    logging.info("Running in Debug Mode")

    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    else:
        port = 8888

    application = tornado.web.Application([
        (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": os.path.join(ROOT_DIR, "static")}),
        ],
        debug=DEBUG_MODE,
        template_path=os.path.join(ROOT_DIR, "templates"))

    application.listen(port)
    tornado.ioloop.IOLoop.instance().start()

