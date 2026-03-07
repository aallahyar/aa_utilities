import textwrap

from aa_utilities.helpers import TextWrapper


text = "line with   space,\n\n\n2nd paragraph with text\n3rd paragraph with a LOOOOOOOOOngWorddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
text_wrapper = textwrap.TextWrapper(width=25)
print(text_wrapper.fill(text))

text_wrapper = textwrap.TextWrapper(width=25, replace_whitespace=False)
print(text_wrapper.fill(text))

text_wrapper = TextWrapper(width=25, keep_newlines=True)
print(text_wrapper(text))