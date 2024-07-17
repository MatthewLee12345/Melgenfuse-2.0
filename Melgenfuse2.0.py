from tkinter import *
from tkinter import ttk
import numpy as np
import mido
from mido import MidiFile, MidiTrack, Message
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
import pygame.mixer

# Predefined melodies for genres
genre_melodies = {
    "classical": [60, 62, 64, 65, 67, 69, 71, 72],
    "jazz": [60, 63, 65, 67, 70, 72, 74, 76],
    "pop": [60, 62, 64, 66, 68, 70, 72, 74],
    "cinematic": [62, 65, 67, 70, 72, 74, 76, 78],
    "rege": [61, 63, 66, 68, 70, 73, 75, 77],
    "folk": [60, 62, 63, 65, 67, 69, 71, 73]
}

melodies = [genre_melodies[key] for key in genre_melodies.keys()]

# Data Preprocessing
input_data = []
output_data = []

for melody in melodies:
    for i in range(len(melody)-2):
        input_data.append(melody[i:i+2])
        output_data.append(melody[i+2])

input_data = np.array(input_data)
output_data = to_categorical(output_data, num_classes=128)

# LSTM Model Setup
model = Sequential()
model.add(LSTM(128, input_shape=(2, 1)))
model.add(Dense(output_data.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Reshape input to be [samples, time steps, features]
input_data = np.reshape(input_data, (input_data.shape[0], 2, 1))

# Training LSTM
model.fit(input_data, output_data, epochs=200, batch_size=10)

def generate_music_lstm(initial_seq):
    generated_melody = initial_seq[:]
    
    for _ in range(20):  # generate 20 more notes for this example
        input_seq = np.reshape(initial_seq, (1, len(initial_seq), 1))
        prediction = np.argmax(model.predict(input_seq, verbose=0))
        generated_melody.append(prediction)
        initial_seq.append(prediction)
        initial_seq = initial_seq[1:]
    
    return generated_melody

def save_as_midi(melody):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    for note in melody:
        track.append(Message('note_on', note=note, velocity=64, time=0))
        track.append(Message('note_off', note=note, velocity=64, time=480))
    
    mid.save('fusion_music.mid')

def compose_music1():
    genre1 = genre_var1.get()
    initial_melody = genre_melodies[genre1][0]
    save_as_midi(generate_music_lstm(initial_melody))
    
def compose_music2():
    genre2 = genre_var2.get()
    initial_melody = genre_melodies[genre2][0]
    save_as_midi(generate_music_lstm(initial_melody))
    
    
def compose_music3():
    genre3 = genre_var3.get()
    initial_melody = genre_melodies[genre3][0]
    save_as_midi(generate_music_lstm(initial_melody))
    
def compose_music():
    genre1 = genre_var1.get()
    genre2 = genre_var2.get()
    genre3 = genre_var3.get()

    # Start with the first note from genre1 and the first note from genre2
    initial_melody = [genre_melodies[genre1][0], genre_melodies[genre2][0]]

    # Optionally, you can also average or somehow blend the notes from the genres:
    # initial_melody = [(genre_melodies[genre1][0] + genre_melodies[genre2][0] + genre_melodies[genre3][0]) // 3]
    fusion_melody = generate_music_lstm(initial_melody)
    save_as_midi(fusion_melody)

    
def play_midi():
    pygame.mixer.init()
    pygame.mixer.music.load('fusion_music.mid')
    pygame.mixer.music.play()
    
# GUI setup
app = Tk()
app.title("Music Fusion LSTM")

genres = ["classical", "jazz", "pop", "cinematic", "rege", "folk"]

genre_var1 = StringVar(app)
genre_var1.set(genres[0])  # default value
genre_dropdown1 = OptionMenu(app, genre_var1, *genres)
genre_dropdown1.config(bg = 'GREEN', fg = "black")
genre_dropdown1.pack()
genre_dropdown1.place(x=400, y=100)

genre_var2 = StringVar(app)
genre_var2.set(genres[1])  # default value
genre_dropdown2 = OptionMenu(app, genre_var2, *genres)
genre_dropdown2.pack()
genre_dropdown2.place(x=400, y=200)

genre_var3 = StringVar(app)
genre_var3.set(genres[2])  # default value
genre_dropdown3 = OptionMenu(app, genre_var3, *genres)
genre_dropdown3.pack()
genre_dropdown3.place(x=400, y=300)


compose1 = Button(app, text="Compose", command=compose_music1)
compose1.pack()
compose1.place(x=500, y=100)

compose2 = Button(app, text="Compose", command=compose_music2)
compose2.pack()
compose2.place(x=500, y=200)

compose3 = Button(app, text="Compose", command=compose_music3)
compose3.pack()
compose3.place(x=500, y=300)

compose = Button(app, text="Compose", command=compose_music)
compose.pack()
compose.place(x=500, y=400)


play_btn = Button(app, text="Play", command=play_midi)
play_btn.pack()
play_btn.place(x=500, y=500)

app.mainloop()