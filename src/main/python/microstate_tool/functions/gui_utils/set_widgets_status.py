"""
This function is designed to modify the state of one or more GUI widgets based on the specified mode.
It can enable, disable, hide, or show a single widget or a list of widgets, as indicated by the mode parameter.

"""
def set_widgets_status(widgets, mode='enable'):
    '''
    widgets: list of widgets, or a single widget
    mode: 'enable', 'disable', 'hide', or 'show'
    '''
    valid_modes = {'enable', 'disable', 'hide', 'show'}

    if mode not in valid_modes:
        print("Invalid mode!")
        return

    if not isinstance(widgets, list):
        widgets = [widgets]

    for item in widgets:
        if mode == 'enable':
            item.setEnabled(True)
        elif mode == 'disable':
            item.setEnabled(False)
        elif mode == 'hide':
            item.hide()
        elif mode == 'show':
            item.show()
