from django import template
register = template.Library()


@register.filter
def get_value(_dict, key):
    return _dict[key]
