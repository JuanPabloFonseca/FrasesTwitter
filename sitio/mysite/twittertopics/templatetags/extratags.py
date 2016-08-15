from django import template
register = template.Library()

@register.filter(name='div')
def div(value, arg):
    try:
        value = int( value )
        arg = int( arg )
        if arg:
            return value / arg
    except: pass
    return ''


@register.filter(name='mul')
def mul(value, arg):
    try:
        value = float( value )
        arg = float( arg )
        if arg:
            return value * arg
    except: pass
    return ''