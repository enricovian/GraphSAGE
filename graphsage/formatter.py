from string import Formatter

class PartialFormatter(Formatter):
    def __init__(self, missing='~', missing_spec=""):
        self.missing = missing
        self.missing_spec = missing_spec

    def get_field(self, field_name, args, kwargs):
        # Handle missing fields
        try:
            return super(PartialFormatter, self).get_field(field_name, args, kwargs)
        except (KeyError, AttributeError):
            return None, field_name

    def format_field(self, value, spec):
        if value is None:
            return super(PartialFormatter, self).format_field(self.missing, self.missing_spec)
        else:
            return super(PartialFormatter, self).format_field(value, spec)
