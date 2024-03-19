
def set_widgets_status(widgets, mode='enable'):
    """
    Modify the state of one or more GUI widgets based on the specified mode.

    Args:
        widgets (list or QWidget): A list of widgets or a single widget.
        mode (str): The mode to apply to the widgets. Valid modes are 'enable', 'disable', 'hide', or 'show'.

    Returns:
        None
    """

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
