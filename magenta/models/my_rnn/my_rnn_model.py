from magenta.models.shared import events_rnn_model


class MelodyRnnModel(events_rnn_model.EventSequenceRnnModel):

    def __init__(self):
        super.__init__(self)