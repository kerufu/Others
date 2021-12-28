import pyttsx3
# check mbrola for more espeak voice: https://github.com/numediart/MBROLA
# put mbrola voice in /usr/share/mbrola

engine = pyttsx3.init() # object creation

# RATE
rate = engine.getProperty('rate')   # get current speaking rate
print(rate)                        # print current voice rate
engine.setProperty('rate', 125)     # set new voice rate


# VOLUME
volume = engine.getProperty('volume')   # get current volume level (min=0 and max=1)
print(volume)                          # printing current volume level
engine.setProperty('volume',1.0)    # set volume level between 0 and 1

# VOICE
voices = engine.getProperty('voices')       # get list of current voices
engine.setProperty('voice', voices[11].id)   # change voice id

engine.say("Hello World!")
engine.say('My current speaking rate is ' + str(rate))
engine.runAndWait()
engine.stop()