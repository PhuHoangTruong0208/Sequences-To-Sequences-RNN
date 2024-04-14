import tf_keras as keras
import tf_keras.backend as K
import numpy as np

class PreprocessingLanguages(keras.layers.Layer):
    def __init__(self, PAD_token: str, START_token: str, END_token: str, OUT_token: str, pad_num=0):
        super(PreprocessingLanguages, self).__init__()
        assert pad_num != 0, "bạn đã thiếu tham số số lượng đệm -> pad_num"
        self.PAD_token = PAD_token
        self.START_token = START_token
        self.END_token = END_token
        self.OUT_token = OUT_token
        self.pad_num = pad_num
    
    def call(self, inputs):
        assert isinstance(inputs, list), "đầu vào phải là 1 danh sách chứa chuổi nguồn và mục tiêu"
        sequence_source, sequence_target = inputs
        
        big_text = ""
        for lines in sequence_source + sequence_target:
            big_text += " "+lines

        vocab = {}
        vocab_idx = list(set(big_text.split()))

        for i in range(len(vocab_idx)):
            vocab[vocab_idx[i]] = i+4
        
        tokens = [self.PAD_token, self.START_token, self.END_token, self.OUT_token]
        for i in range(len(tokens)):
            vocab[tokens[i]] = i
        
        # từ vựng ngược
        rev_vocab = {v: k for k, v in vocab.items()}

        for i in range(len(sequence_target)):
            if len(sequence_target[i].split()) > self.pad_num:
                sequence_target[i] = " ".join(sequence_target[i].split()[:self.pad_num])
            sequence_target[i] = f"{self.START_token} {sequence_target[i]} {self.END_token}"

        # mã hóa sequences theo thứ tự từ vựng
        for i in range(len(sequence_source)):
                sequence_source[i] = [vocab[w] for w in sequence_source[i].split()]
        for i in range(len(sequence_target)):
                sequence_target[i] = [vocab[w] for w in sequence_target[i].split()]
        
        # đệm câu
        sequence_source = keras.preprocessing.sequence.pad_sequences(sequences=sequence_source,
                                                                     maxlen=self.pad_num, padding="post",
                                                                     truncating="post")
        sequence_target = keras.preprocessing.sequence.pad_sequences(sequences=sequence_target,
                                                                     maxlen=self.pad_num, padding="post",
                                                                     truncating="post")
        sequence_target_predict = []
        for lines in sequence_target:
            sequence_target_predict.append(lines[1:])
        sequence_target_predict = keras.preprocessing.sequence.pad_sequences(sequences=sequence_target_predict,
                                                                     maxlen=self.pad_num, padding="post",
                                                                     truncating="post")
        return sequence_source, sequence_target, sequence_target_predict, vocab, rev_vocab


class Attention(keras.layers.Layer):
    def __init__(self):
        super(Attention, self).__init__()

    def build(self, input_shape):
        # các mạng tính trọng số chú ý
        self.W1 = keras.layers.Dense(units=input_shape[0][2], use_bias=False)
        self.W2 = keras.layers.Dense(units=input_shape[0][2], use_bias=False)
        self.V = keras.layers.Dense(units=1, use_bias=False)
        # các hàm kích hoạt tính điểm chú ý
        self.ActivationTanh = keras.layers.Activation(activation="tanh")
        self.ActivationSoftmax = keras.layers.Activation(activation="softmax")
        return super(Attention, self).build(input_shape)
    
    def call(self, inputs):
        assert isinstance(inputs, list), "đầu vào phải là 1 danh sách chứa lần lượt encoder output và decoder output"
        Q_inputs, V_inputs = inputs
        
        # tính toán trọng số chú ý
        def compute_attention_weights(inputs, states):
            W1 = self.W1(Q_inputs)
            W2 = K.expand_dims(self.W2(inputs), 1)
            scores = self.ActivationTanh(W1+W2)
            attention_weights = K.squeeze(self.V(scores), -1)
            attention_weights = self.ActivationSoftmax(attention_weights)
            return attention_weights, [attention_weights]
        
        # tính toán kết quả chú ý
        def compute_attention_output(inputs, states):
            attention_output = K.expand_dims(inputs, -1) * Q_inputs
            attention_output = K.sum(attention_output, 1)
            return attention_output, [attention_output]
        
        # các trạng thái tương ứng cho rnn
        states_attention_weights = K.sum(Q_inputs, 2)
        states_attention_output = K.sum(Q_inputs, 1)
        
        # dùng rnn để tạo ra 1 kết quả trọng số chú ý có shape tương đương với encoder out
        _ ,attention_weights, _ = K.rnn(
            compute_attention_weights, V_inputs, [states_attention_weights]
            )
        _, attention_output, _ = K.rnn(
            compute_attention_output, attention_weights, [states_attention_output]
        )
        return attention_output, attention_weights
    
    
class Seq2SeqModel(keras.layers.Layer):
    def __init__(self, epochs=0, pad_num=0, vocab_size=0, dropout=0.05, units=200, trainable=True, name=None,
                  dtype=None, dynamic=False, **kwargs):
        super(Seq2SeqModel, self).__init__(trainable, name, dtype, dynamic, **kwargs)
        assert vocab_size != 0 and pad_num != 0 and epochs != 0, "bạn đã thiếu 1 trong 3 tham số sau, vui lòng xem lại: vocab_size, pad_num, epochs"
        self.units = units
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.pad_num = pad_num
        self.epochs = epochs

    def build(self, input_shape):
        en_input_shape = keras.layers.Input(shape=(self.pad_num,))
        de_input_shape = keras.layers.Input(shape=(self.pad_num,))
        self.en_input_shape = en_input_shape
        self.de_input_shape = de_input_shape

        self.Embed_Encoder = keras.layers.Embedding(input_dim=self.vocab_size+1, output_dim=self.units,
                                                        trainable=True)(en_input_shape)
        self.Embed_Decoder = keras.layers.Embedding(input_dim=self.vocab_size+1, output_dim=self.units,
                                                        trainable=True)(de_input_shape)
        
        self.RNN_Encoder = keras.layers.LSTM(units=self.units, return_sequences=True, return_state=True,
                                             use_bias=True, activation="tanh", dropout=self.dropout)
        self.RNN_Decoder = keras.layers.LSTM(units=self.units, return_sequences=True, return_state=True,
                                             use_bias=True, activation="tanh", dropout=self.dropout)
        
        self.Attention = Attention()
        self.Concat = keras.layers.Concatenate(axis=-1)
        self.DensePredict = keras.layers.Dense(units=self.vocab_size, activation="softmax")

        return super(Seq2SeqModel, self).build(input_shape)
    
    def call(self, inputs):
        assert isinstance(inputs, list), "đầu vào phải là 1 danh sách chứa lần lượt X -> y -> y predict"
        X, y, y_p = inputs
        
        def encoder_RNN():
            encoder_output, hidden, cell = self.RNN_Encoder(self.Embed_Encoder)
            return encoder_output, [hidden, cell]
        
        def decoder_RNN(encoder_states):
            decoder_output, hidden, cell = self.RNN_Decoder(self.Embed_Decoder,
                                                            initial_state=encoder_states)
            return decoder_output, [hidden, cell]
        
        def attention(inputs):
            assert isinstance(inputs, list), "đầu vào phải là 1 danh sách chứa lần lượt encoder output và decoder output"
            attention_output, attention_weights = self.Attention(inputs)
            attention_output = self.Concat([inputs[1], attention_output])
            return attention_output, attention_weights
        
        def dense_predict(attention_output):
            return self.DensePredict(attention_output)
        
        encoder_output, encoder_states = encoder_RNN()
        decoder_output, decoder_states = decoder_RNN(encoder_states)
        attention_output, attention_weights = attention([encoder_output, decoder_output])
        dense_predict_output = dense_predict(attention_output)

        # đào tạo mô hình
        seq2seq_model = keras.models.Model([self.en_input_shape, self.de_input_shape],
                                           dense_predict_output)
        seq2seq_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                              loss="sparse_categorical_crossentropy",
                              metrics=['accuracy'])
        seq2seq_model.fit([X, y], y_p, epochs=self.epochs)

        # mô hình suy luận
        states_input = [keras.layers.Input(self.units,), keras.layers.Input(self.units,)]
        encoder_inference = keras.models.Model(self.en_input_shape, [encoder_output, encoder_states])
        decoder_input, hidden, cell = self.RNN_Decoder([self.Embed_Decoder] + states_input)
        decoder_states_input = [hidden, cell]
        decoder_inference = keras.models.Model([self.de_input_shape] + states_input,
                                               [decoder_input] + decoder_states_input)
        return encoder_inference, decoder_inference
    
# dữ liệu mẫu
with open("data.txt", "r", encoding="utf-8") as file:
    data = file.read().splitlines()
X, y= [], []
for i in range(len(data)):
    try:
        y.append(data[i+1])  
    except:
        break
    X.append(data[i])

# siêu tham số mẫu để test
pad = 10
hid_num = 10
epochs = 1

preprocessing = PreprocessingLanguages(pad_num=pad,START_token="<start>", END_token="<end>", PAD_token="<pad>",
                             OUT_token="<out>")
seq_s, seq_t, seq_t_p, vocab, rev_vocab = preprocessing([X, y])

seq2seq = Seq2SeqModel(vocab_size=len(vocab), epochs=epochs, units=hid_num, pad_num=pad)
en_inf, de_inf = seq2seq([seq_s, seq_t, seq_t_p])

while True:
    user_input = input("Bạn: ")
    vector_input = []
    for word in user_input.split():
        try:
            vector_input.append(vocab[word])
        except:
            vector_input.append(vocab["<out>"])
    vector_input = [vector_input]
    vector_input = keras.preprocessing.sequence.pad_sequences(sequences=vector_input, maxlen=pad, padding="post", truncating="post")

    en_op, en_hid = en_inf.predict(vector_input)
    target = np.zeros((1,1))
    target[0,0] = vocab["<start>"]
    
    sentence = ""
    for i in range(20):
        de_op, de_hid, de_cell = de_inf.predict([target] + en_hid, verbose=0)
        att_op, att_hid = seq2seq.Attention([en_op, de_op])
        concat_op = seq2seq.Concat([de_op, att_op])
        softmax = seq2seq.DensePredict(concat_op)
        clsf_op = np.argmax(softmax)
        word_op = rev_vocab[clsf_op]

        if word_op in "<end>":
            break

        sentence += word_op+" "

        en_hid = [de_hid, de_cell]
        target[0,0] = clsf_op
    
    print("Model:", sentence)