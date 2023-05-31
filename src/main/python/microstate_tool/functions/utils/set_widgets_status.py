def set_widgets_status(widgets, enable=True):
        '''
        widgets: list of widgets, or a single widget
        enable: if True, setEnabled; if False, setDisabled.
        '''
        if type(widgets) == list:
            for item in widgets:
                item.setEnabled(True) if enable else item.setDisabled(True)
        else:
            widgets.setEnabled(True) if enable else widgets.setDisabled(True)