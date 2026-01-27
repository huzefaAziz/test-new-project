class Meta(type):
    def __new__(mcls, name, bases, namespace):
        cls = super().__new__(mcls, name, bases, namespace)
        
        def spawn(level=0):
            new_name = f"Class_{level}"
            
            return Meta(new_name, (), {
                "level": level,
                "spawn": staticmethod(lambda l=level+1: spawn(l))
            })
        
        cls.spawn = staticmethod(spawn)
        return cls


class Origin(metaclass=Meta):
    pass


# Infinite class generation
C = Origin.spawn()
D = C.spawn()
E = D.spawn()
F = E.spawn()
class Meta(type):
    def __new__(mcls, name, bases, namespace):
        cls = super().__new__(mcls, name, bases, namespace)
        
        def spawn(level=0):
            new_name = f"Class_{level}"
            
            return Meta(new_name, (), {
                "level": level,
                "spawn": staticmethod(lambda l=level+1: spawn(l))
            })
        
        cls.spawn = staticmethod(spawn)
        return cls


class Origin(metaclass=Meta):
    pass


# Infinite class generation
C = Origin.spawn()
D = C.spawn()
E = D.spawn()
F = E.spawn()
