Hello! Thank you for the interest in my project.
This is Yerassyl Abeuov’s project for Infomatrix Asia 2026 in AI programming (“Neuromusix: Adaptive Music Generation Using Reinforcement Learning”)

Neuromusix is an AI-based music generation application (written in Python) that creates original melodies based on user preferences and emotional profiles. The program analyzes musical features such as energy, complexity, depth, openness, and structure, then generates personalized compositions using music theory and algorithmic rules. Users can listen to tracks, visualize melodies, give feedback (like/dislike), and export music in MIDI, WAV, or notation formats. Over time, Neuromusix adapts to the user’s taste using reinforcement learning, making each new composition more personalized.
The manual is in the manual.txt file that can be accessed through the application.
The attribution to the soundfont is in the manual.


  To run the program, install these libraries (in venv) by inserting this in a command line of a Python interpretator:
        pip install -r requirements.txt
  for macOS, you may need:
        python3 -m pip install -r requirements.txt

IMPORTANT NOTE!!! https://drive.google.com/file/d/1-CtSEzzo5ECzvRppC5jtlmoorHVKjlpA/view?usp=sharing
Sorry. You have to download the file from the link above and add it to the folder of the code if you want the program to export in WAV properly. The file is too big for GitHub. It adds synthesizer for WAV. 
Also, you need to install Fluidsynth for audio export.

To install Fluidsynth for exporting in WAV, you may use this:
-   MacOS: brew install fluidsynth
-   Windows: installing FluidSynth from their website and adding to the path of the program
-   Linux: sudo apt install fluidsynth

To export as WAV, FluidSynth must be installed on a computer. 
To export as XML, MuseScore 4 must be installed on a computer.

Attribution: Sounfont GeneralUser GS 1.471 by Schristian Collinss - https://schristiancollins.com/generaluser.php
