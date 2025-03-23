import _pickle as cPickle
from pathlib import Path
from atexit import register
from time import sleep

from llmag.llmag_utils.log_utils import *

def saveCache(cache, cfile):
    '''
    Function saving CachedFunction's cache - made for registering
    it for execution at program exit.
    '''
    if cache:
        cPickle.dump(cache, open(cfile, 'wb'))

class CachedFunction():
    #todo synchronization in the case two CachedCohs write to same file in parallel

    def __init__(self, function, cacheFolder='.', saveEvery=50, verbose=False):
        '''
        :param function: callable with .id property
        :param cacheFolder: file for saving computed values
        :param saveEvery: save cache for saveEvery new coherence calculations
                    cache is always saved when object is destroyed
        '''
        self.cacheFolder = Path(cacheFolder)
        if not self.cacheFolder.exists(): self.cacheFolder.mkdir(exist_ok=True)
        self.func_id = function.id if hasattr(function, 'id') else str(function.__name__)
        self.cacheFile = path.join(cacheFolder, self.__hid(self.func_id) + '.pickle')
        self.__cache = None
        self.__log = createLogger(fullClassName(self), INFO)
        self.function = function
        self.saveEvery = saveEvery
        self.newCalcCnt = 0
        if hasattr(function, 'measure'):
            self.measure = function.measure
        if verbose: self.logCacheInitMessage()
        #print self.__cacheInitMessage()

    @property
    def id(self): return self.func_id

    def __cacheInitMessage(self):
        msg = 'Cache Function Loaded\n'
        msg += 'function id: %s\n' % self.func_id
        msg += 'cache folder: %s\n' % self.cacheFolder
        msg += 'cache file: %s\n' % (self.__hid(self.func_id) + '.pickle')
        return msg

    def logCacheInitMessage(self):
        self.__log.info(self.__cacheInitMessage())

    def __hid(self, id):
        '''
        Since id can be used to create a file with the name corresponding to id,
        and ids can be longer than max. allowed file size, this function
        produces a hash of a string id, that is shorter then max. file size.
        Hash is reproducible across runs and machines.
        Possibilities of collision should be astronomically small.
        '''
        from hashlib import pbkdf2_hmac
        h = pbkdf2_hmac('sha512', str(id).encode('utf-8'),
                        str(self.__class__.__name__).encode('utf-8'), 10000, dklen=50)
        hid = 'hid'+''.join('%d'%b for b in h)
        return hid

    def __call__(self, *args, **kwargs):
        '''
        :param words: list of whitespace separated strings
        :return:
        '''
        pid = self.__paramId(*args, **kwargs)
        rtrn_val = self.__load(pid)
        if rtrn_val is None:
            rtrn_val = self.function(*args, **kwargs)
            self.__save(pid, rtrn_val)
        return rtrn_val

    def is_cached(self, *args, **kwargs):
        '''
        check if the value for the given params exists in the cache.
        '''
        pid = self.__paramId(*args, **kwargs)
        return self.__load(pid) is not None

    def __paramId(self,  *args, **kwargs):
        '''
        Return unique string id composed of arguments, which can
         be used as an id for querying and saving to cache.
        '''
        def v2s(obj):
            '''value 2 string'''
            import types
            if obj == None: return None
            if hasattr(obj, 'id'):
                return obj.id
            else:
                if hasattr(obj, '__name__'):
                    return obj.__name__
                else:
                    return str(obj)
        astr = ','.join( v2s(a) for a in args )
        kwastr = ','.join( '%s:%s'%(v2s(k), v2s(v)) for k, v in kwargs.items() )
        id_ = 'ARGS[%s]_KWARGS[%s]'%(astr, kwastr)
        return id_

    def __load(self, pid):
        if self.__cache is None: self.__loadCreateCache()
        return self.__cache[pid] if pid in self.__cache else None

    def __loadCreateCache(self):
        '''create cache or load it from file'''
        if path.exists(self.cacheFile):
            self.__cache = cPickle.load(open(self.cacheFile, 'rb'))
        else:
            self.__cache = {}
        # register save at program exit since garbage collect is not guaranteed to happen
        register(saveCache, self.__cache, self.cacheFile)

    def __save(self, pid, coh):
        if pid in self.__cache: return
        self.__cache[pid] = coh
        self.newCalcCnt += 1
        if self.newCalcCnt % self.saveEvery == 0:
            self.saveCache()
            self.newCalcCnt = 0

    def saveCache(self):
        if self.__cache and self.newCalcCnt > 0:
            cPickle.dump(self.__cache, open(self.cacheFile, 'wb'))

    def __del__(self):
        self.saveCache()
        if self.__cache is not None: self.__cache.clear()

    @staticmethod
    def unite(target, sources, checkEquality=True):
        '''
        Add all the param->value mappings from sources cached functions
        to targed cached function and save the result.
        :param target: cached function
        :param sources: list of cached functions
        :param checkEquality: if true, check equality of values for duplicated source keys
        :return:
        '''
        if not isinstance(target, CachedFunction): return
        for s in sources:
            if not isinstance(s, CachedFunction): return
        target.__loadCreateCache()
        for s in sources:
            s.__loadCreateCache()
            if checkEquality:
                for k, v in s.__cache.iteritems():
                    if k in target.__cache:
                        assert target.__cache[k] == s.__cache[k]
            target.__cache.update(s.__cache)
            print(len(s.__cache))
        target.newCalcCnt = 1 # without, it will not save
        target.saveCache()

def testHash():
    from hashlib import pbkdf2_hmac
    for p in ['a long id1', 'a long id2']:
        h = pbkdf2_hmac('sha512', p.encode('utf-8'),
                        str(CachedFunction.__name__).encode('utf-8'), 10000, dklen=50)
        print(type(h), len(h))
        hid = 'hid' + ''.join('%d' % b for b in h)
        print(hid, len(hid))
    #for i in h: print type(i[0]), ord(i)

def testCachedFunction():
    ''' Run twice, second, cached, run should finish instantenously. '''
    save_folder = 'cache_test'
    def test_fn(x):
        sleep(0.5)
        return str(x)+str(x)
    cfn = CachedFunction(test_fn, cacheFolder=save_folder, saveEvery=1, verbose=True)
    for i in range(10):
        val = cfn(i)
        print(f'arg: {i}, return: {val}')

if __name__ == '__main__':
    #testHash()
    testCachedFunction()