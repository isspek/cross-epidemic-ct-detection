from abc import abstractmethod

from llmag.common.Identifiable import Identifiable


class Text(Identifiable):
    '''
    Defines interface for Text-like objects, and provides a simple implementation.
    'id' (unique id at corpus level) and 'text' (text as string) are
    mandatory attributes, other attributes are text-specific and optional.
    '''

    @property
    @abstractmethod
    def text(self):
        ''' String representation of the text. '''

    def __str__(self): return self.text

    @abstractmethod
    def __iter__(self):
        '''
        Iteration over name, value pairs for non-standard (id, text) attributes in the text.
        '''

class SimpleText(Text):
    '''
    Simple implementation for Text, 'id' (unique id at corpus level) and 'text' (text as string)
    are mandatory attributes, other attributes are text-specific and optional.
    '''

    @property
    def id(self): return self._id

    @property
    def text(self): return self._text

    def __init__(self, id, text, **attributes):
        '''
        Define id, text and a list of arbitrary attributes.
        :param attributes: list of name=value, converted to object attributes
        '''
        self._id = id
        self._text = text
        for attr, val in attributes.items():
            self.__dict__[attr] = val

    def __iter__(self):
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                if key != 'id' and key != 'text':
                    yield key, value

def copy_text(txt):
    '''
    Create a shallow copy of a Text-like object, as a new Text object with identical attributes.
    '''
    atts = { name:val for name, val in txt }
    return Text(txt.id, txt.text, **atts)

if __name__ == '__main__':
    txt = SimpleText(123, 'a text.', value=0)
    print(txt)