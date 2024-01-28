from abc import ABC

class Module(ABC):
    
    def forward(self, *ar, **kw):
        raise NotImplementedError
    
    def __call__(self, *ar, **kw):
        return self.forward(*ar, **kw)