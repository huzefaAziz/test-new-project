def MetaFactory(depth=0):
    class RecursiveMeta(type):
        def __new__(mcls, name, bases, namespace):
            namespace.setdefault('depth', depth)
            cls = super().__new__(mcls, name, bases, namespace)
            return cls
    
    # Add a descriptor for lazy Next creation
    class NextDescriptor:
        def __get__(self, obj, objtype):
            # Check if Next is already cached
            if not hasattr(objtype, '_Next_cache'):
                if depth < 10**9:
                    NewMeta = MetaFactory(depth + 1)
                    Next = NewMeta('Next', (), {})
                    objtype._Next_cache = Next
                else:
                    objtype._Next_cache = None
            return objtype._Next_cache
    
    RecursiveMeta.Next = NextDescriptor()
    
    return RecursiveMeta

class Genesis(metaclass=MetaFactory()):
    pass

X = Genesis.Next.Next.Next.Next
print(X.depth) 