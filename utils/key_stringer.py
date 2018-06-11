class KeyStringer(object):

    def __init__(self, env):
        better_buttons = env.unwrapped.buttons[:] # clone
        index_of_none = better_buttons.index(None)
        better_buttons[index_of_none] = "_"
        self.short_names = [button[0] for button in better_buttons]
        self.order_upper = [name.upper() for name in order]

    def keys_to_string(self, keys):
        accu = ""
        for i, key in enumerate(keys):
            accu += self.short_names[i] if key else "."
        return accu

    def key_dict_to_fm2(self, key_dict):
        accu = "|0|"

        for name in self.order_upper:
            accu += name[0] if key_dict[name] else "."
        accu += "|||"
        return accu