from __future__ import with_statement, absolute_import
import functools
import logging
import os
import simplejson
import sys
import threading

import tornado
import tornado.ioloop
import tornado.web

import rpcs
DEBUG_MODE = False

# XXX: location samples device id-> list of scanResults
samples = {}

class RpcRequestHandler(tornado.web.RequestHandler):
    RPC_METHODS = {
        "location_sample" : rpcs.location_sample,
        }

    def __handle_request(self, args):
        try:
            args = json_decode(args)
            args = dict((str(k), v) for k, v in args.iteritems())

            method = args.pop("method")
        except Exception:
            self.send_error(400)

        if not method in RpcRequestHandler.RPC_METHODS:
            self.send_error(404)

        logging.info("RPC[%s]: args: %s" % (method, simplejson.dumps(args, sort_keys=True)))
        self.__rpc_method = method
        thread = threading.Thread(target = self.__async_rpc,
                kwargs = {
                    'method' : method,
                    'kwargs' : args,
                    })
        thread.start()

    def __async_rpc(self, method, kwargs):
        try:
            ret = RpcRequestHandler.RPC_METHODS[method](**kwargs)
            ret = simplejson.dumps(ret)
            tornado.ioloop.IOLoop.instance().add_callback(
                    self.async_callback(functools.partial(self.__finish_request, ret)))
        except:
            # explicitly catch any type of exception here, transfer code to main thread and then reraise it
            tornado.ioloop.IOLoop.instance().add_callback(
                    self.async_callback(functools.partial(self.__error_request, sys.exc_info())))

    def __error_request(self, exc_info):
        raise exc_info[0], exc_info[1], exc_info[2]

    def __finish_request(self, ret):
        self.write(ret)
        self.finish()

    @tornado.web.asynchronous
    def post(self):
        return self.__handle_request(self.request.body)

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
        (r"/rpc", RpcRequestHandler),
        ],
        debug=DEBUG_MODE,
        template_path=os.path.join(ROOT_DIR, "templates"))

    application.listen(port)
    tornado.ioloop.IOLoop.instance().start()

