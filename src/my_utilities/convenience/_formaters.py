import pprint


class PrettyPrinter():
    def __init__(
            self, 
            indent=' ' * 4, 
            sep=',', 
            display_width=80, 
            default=str,
            max_n_columns=200,
            max_n_rows=20,
        ):
        self.indent = indent
        self.sep = sep
        self.default = default
        self.pp = pprint.PrettyPrinter()
        self.display_width = display_width
        self.max_n_columns = max_n_columns
        self.max_n_rows = max_n_rows
    
    def pprint():
        pass