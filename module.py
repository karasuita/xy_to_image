import sys
import os
import inspect
from inspect import Parameter
from collections import OrderedDict
import json
import tempfile
import zipfile
import torch


class InitParams:

    def __init__(self, f_locals=None, signature=None):
        self._args = OrderedDict()
        self._vargs = list()
        self._kwargs = OrderedDict()

        self._init_args(f_locals, signature)

    def _init_args(self, f_locals=None, signature=None):
        from inspect import Parameter

        if f_locals is None or signature is None:
            return

        args = OrderedDict()
        vargs = list()
        kwargs = OrderedDict()

        for i, name in enumerate(signature.parameters):
            # pass self
            if i == 0:
                continue
            parameter = signature.parameters[name]
            kind = parameter.kind

            if kind == Parameter.POSITIONAL_ONLY or kind == Parameter.POSITIONAL_OR_KEYWORD:
                if name in f_locals:
                    value = f_locals[name]
                elif parameter.default is not Parameter.empty:
                    value = parameter.default
                else:
                    value = None
                args[name] = value
            elif kind == Parameter.VAR_POSITIONAL:
                if name in f_locals:
                    vargs.extend(f_locals[name])
            elif kind == Parameter.KEYWORD_ONLY:
                if name in f_locals:
                    value = f_locals[name]
                elif parameter.default is not Parameter.empty:
                    value = parameter.default
                else:
                    value = None
                kwargs[name] = value
            elif kind == Parameter.VAR_KEYWORD:
                if name in f_locals:
                    kwargs.update(f_locals[name])

        self._args = args
        self._vargs = vargs
        self._kwargs = kwargs

    @property
    def args(self):
        return self._args

    @property
    def vargs(self):
        return self._vargs

    @property
    def kwargs(self):
        return self._kwargs

    def to_json(self, indent=None):
        import json
        return json.dumps({
            'args': self._args,
            'vargs': self._vargs,
            'kwargs': self._kwargs,
        }, indent=indent)


class SaveInitParams:

    def __init__(self):
        super().__init__()
        frame = sys._getframe(1)
        klass = self.__class__
        f_locals = None
        while frame is not None:
            if '__class__' in frame.f_locals and frame.f_locals['__class__'] is klass:
                f_locals = frame.f_locals
                break
            frame = frame.f_back
        signature = inspect.signature(klass.__init__)
        self._init_params = InitParams(f_locals, signature)

    @property
    def init_params(self):
        return self._init_params


class Module(SaveInitParams, torch.nn.Module):

    SETTINGS_NAME = 'settings.json'
    MODEL_NAME = 'model.pth'

    @classmethod
    def load(cls, filepath):
        with zipfile.ZipFile(filepath) as zf:
            with zf.open(Module.SETTINGS_NAME) as f:
                settings = json.load(f)

            with tempfile.TemporaryDirectory() as dirname:
                model_path = zf.extract(Module.MODEL_NAME, dirname)
                        
                with open(model_path, 'rb') as temp_f:
                    state_dict = torch.load(temp_f)

        signature = inspect.signature(cls.__init__)

        args = list()
        kwargs = dict()
        for i, name in enumerate(signature.parameters):
            if i == 0:
                continue #self
            parameter = signature.parameters[name]
            kind = parameter.kind

            if kind == Parameter.POSITIONAL_ONLY or kind == Parameter.POSITIONAL_OR_KEYWORD:
                if name in settings['args']:
                    value = settings['args'][name]
                elif parameter.default is not Parameter.empty:
                    value = parameter.default
                else:
                    value = None
                args.append(value)
            elif kind == Parameter.KEYWORD_ONLY:
                if name in settings['kwargs']:
                    value = settings['kwargs'][name]
                    del settings['kwargs'][name]
                elif parameter.default is not Parameter.empty:
                    value = parameter.default
                else:
                    value = None
                kwargs[name] = value
        
        if settings['vargs']:
            args.extend(settings['vargs'])

        if settings['kwargs']:
            kwargs.update(settings['kwargs'])

        model = cls(*args, **kwargs)
        model.load_state_dict(state_dict)

        return model


    def save(self, filepath):
        with tempfile.TemporaryDirectory() as dirname:
            json_path = os.path.join(dirname, Module.SETTINGS_NAME)
            with open(json_path, 'w') as f:
                f.write(self.init_params.to_json(indent=2))

            model_path = os.path.join(dirname, Module.MODEL_NAME)
            torch.save(self.state_dict(), model_path)

            with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.write(json_path, Module.SETTINGS_NAME)
                zf.write(model_path, Module.MODEL_NAME)
