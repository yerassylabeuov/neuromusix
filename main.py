#source venv/bin/activate

import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import random
import os
import platform
import json
import pygame
from music21 import stream, note, chord, tempo, key, roman, instrument, environment, pitch
import threading
import subprocess  
from shutil import which 
from PIL import Image, ImageTk  
SOUNDFONT_PATH = "soundfont.sf2" 

us = environment.UserSettings()
us['musicxmlPath'] = '/Applications/MuseScore 4.app/Contents/MacOS/mscore'
us['musescoreDirectPNGPath'] = '/Applications/MuseScore 4.app/Contents/MacOS/mscore'

INSTRUMENTS = {
    "Piano": instrument.Piano,
    "Electric Piano": instrument.ElectricPiano,
    "Organ": instrument.PipeOrgan,
    "Marimba": instrument.Marimba,
    "Xylophone": instrument.Xylophone,
    "Acoustic Guitar": instrument.AcousticGuitar,
    "Electric Guitar": instrument.ElectricGuitar,
    "Violin": instrument.Violin,
    "Violoncello": instrument.Violoncello,
    "Flute": instrument.Flute,
    "Clarinet": instrument.Clarinet,
    "Trumpet": instrument.Trumpet,
    "Saxophone": instrument.AltoSaxophone,
    "Choir": instrument.Choir,
    "Koto": instrument.Koto,
    "Vocalist": instrument.Vocalist,
    "Pad": instrument.Ukulele
}

INSTRUMENT_ICONS = {
    "Piano": "🎹",
    "Electric Piano": "🎹⚡",
    "Organ": "⛪",
    "Marimba": "🪵",
    "Xylophone": "🦴",
    "Acoustic Guitar": "🎸",
    "Electric Guitar": "🎸⚡",
    "Violin": "🎻",
    "Violoncello": "🎻⬇️",
    "Flute": "🪈",
    "Clarinet": "🎷⬛",
    "Trumpet": "🎺",
    "Saxophone": "🎷",
    "Choir": "🎹☁️",
    "Koto": "🇯🇵",
    "Vocalist": "🎤",
    "Pad": "🎛️"
}

COLORS = {
    "bg_main": "#121212",  
    "bg_panel": "#1E1E1E",    
    "text_main": "#E0E0E0", 
    "text_dim": "#888888",     
    "highlight": "#3498DB",    
    
    "shadow": "#080808",        

    "btn_base": "#25303B",      
    "btn_hover": "#34495E",     
    
    "btn_green": "#1E8449",    
    "btn_green_hover": "#27AE60",
    
    "btn_red": "#922B21",      
    "btn_red_hover": "#C0392B",
    
    "btn_blue": "#1F618D",     
    "btn_blue_hover": "#2980B9",
    "graph_bg": "#2B2B2B",    
    "graph_line": "#3498DB",
    "text_success": "#58D68D", 
    "text_error": "#EC7063"     
}

class ShadowButton(tk.Frame):
    def __init__(self, parent, text, command=None, 
                 width=20, height=2, 
                 bg_color=COLORS["btn_base"], fg_color=COLORS["text_main"], 
                 hover_color=COLORS["btn_hover"], 
                 shadow_color=COLORS["shadow"],
                 font=("Arial", 11, "bold")):
        
        super().__init__(parent, bg=shadow_color)
        
        self.command = command
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.fg_color = fg_color
        self.is_disabled = False

        self.lbl = tk.Label(self, text=text, bg=bg_color, fg=fg_color, font=font, 
                            width=width, height=height, cursor="hand2")
        
        self.lbl.pack(fill=tk.BOTH, expand=True, padx=(0, 3), pady=(0, 3))
        
        self.lbl.bind("<Button-1>", self.on_click)
        self.lbl.bind("<Enter>", self.on_enter)
        self.lbl.bind("<Leave>", self.on_leave)

    def on_click(self, event):
        if self.command and not self.is_disabled:
            self.lbl.pack_configure(padx=(1, 0), pady=(1, 0))
            self.update_idletasks()
            self.after(100, lambda: self.lbl.pack_configure(padx=(0, 3), pady=(0, 3)))
            self.command()

    def on_enter(self, event):
        if not self.is_disabled: self.lbl.configure(bg=self.hover_color)
    def on_leave(self, event):
        if not self.is_disabled: self.lbl.configure(bg=self.bg_color)
    def config_state(self, state="normal"):
        if state == "disabled":
            self.is_disabled = True
            self.lbl.configure(fg="#555", cursor="arrow")
        else:
            self.is_disabled = False
            self.lbl.configure(fg=self.fg_color, cursor="hand2")

class MusicAnalyst:
    @staticmethod
    def analyze_stream(music_stream):
        features = {}
    
        try:
            melody_part = music_stream.parts[0] if music_stream.parts else music_stream
        except:
            melody_part = music_stream
            
        flat_notes = melody_part.flatten().notes

        #energy
        bpm = 120
        try:
            mm = music_stream.metronomeMarkBoundaries()
            if mm: bpm = mm[0][2].number
        except: pass
        
        norm_bpm = min(max((bpm - 60) / 120, 0.0), 1.0)
        
        if len(flat_notes) > 0:
            total_duration = music_stream.duration.quarterLength
            if total_duration == 0: total_duration = 1
            density = len(flat_notes) / total_duration
            norm_density = min(density / 2.0, 1.0)
        else:
            norm_density = 0.5

        features['energy'] = (norm_bpm * 0.6) + (norm_density * 0.4)

        #depth
        pitches = [n.pitch.midi for n in flat_notes if n.isNote]
        if len(pitches) >= 2:
            pitch_range = max(pitches) - min(pitches)
            mean_pitch = sum(pitches) / len(pitches)
            range_score = min(pitch_range / 24.0, 1.0)
            center_distance = abs(mean_pitch - 60)
            center_score = max(1.0 - center_distance / 24.0, 0.0)
            intervals = [abs(pitches[i] - pitches[i-1]) for i in range(1, len(pitches))]
            avg_interval = sum(intervals) / len(intervals)
            interval_score = min(avg_interval / 7.0, 1.0)

            features['depth'] = min(0.4 * range_score + 0.4 * center_score + 0.2 * interval_score, 1.0)
        else:
            features['depth'] = 0.3

        #openness
        melody_pitches = [p.ps for p in melody_part.flatten().pitches]
        if melody_pitches:
            p_range = max(melody_pitches) - min(melody_pitches)
            features['openness'] = min(p_range / 48.0, 1.0)
        else:
            features['openness'] = 0.5

        #complexity
        intervals = []
        if len(pitches) > 1:
            for i in range(len(pitches)-1):
                intervals.append(abs(pitches[i+1] - pitches[i]))
            avg_interval = sum(intervals)/len(intervals) if intervals else 0
            features['complexity'] = min(avg_interval / 7.0, 1.0)
        else:
            features['complexity'] = 0.0

        #structure
        from collections import Counter
        durations = [round(n.duration.quarterLength, 2) for n in flat_notes]
        if durations:
            counts = Counter(durations)
            most_common_ratio = counts.most_common(1)[0][1] / len(durations)
            features['structure'] = min(max(most_common_ratio * 1.4, 0.0), 1.0)
        else:
            features['structure'] = 0.5

        return features

    @staticmethod
    def profile_distance(track_features, user_profile):
        keys = ['energy', 'depth', 'complexity', 'openness', 'structure']
        diffs = []
        for k in keys:
            diffs.append(abs(track_features.get(k, 0.5) - user_profile.get(k, 0.5)))
        return sum(diffs) / len(diffs)

class UserProfile:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.filename = os.path.join(base_dir, "user_profile.json")
        
        self.learning_rate = 0.4
        self.dislike_factor = 0.7
        self.profile = {
            'energy': 0.5, 'depth': 0.5, 'complexity': 0.5,
            'openness': 0.5, 'structure': 0.5
        }
        self.load_profile()

    def update_profile(self, track_features, rating):
        for key in self.profile:
            current_val = self.profile[key]
            track_val = track_features.get(key, 0.5)
            diff = track_val - current_val
            if rating > 0:
                delta = diff * self.learning_rate
            else:
                delta = -diff * self.learning_rate * self.dislike_factor
            
            delta += 0.2 * (1 if delta > 0 else -1) * abs(diff)
            new_val = current_val + delta
            self.profile[key] = max(0.01, min(new_val, 0.99))
        
        self.save_profile() 
        print(f"[Profile updated] {self.profile}")

    def save_profile(self):
        try:
            with open(self.filename, 'w') as f:
                json.dump(self.profile, f, indent=4)
            print(f"--> JSON saved: {self.filename}")
        except Exception as e:
            print(f"SAVE ERROR: {e}")
            try:
                messagebox.showerror("ERROR", f"Profile could not be saved:\n{e}")
            except: pass

    def load_profile(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    self.profile = json.load(f)
            except:
                pass
        else:
            print("New profile is created.")

    def get_vector(self):
        return [self.profile['energy'], self.profile['depth'], self.profile['complexity'], 
                self.profile['openness'], self.profile['structure']]

class MelodyGenerator:
    def inject_creativity(self, profile, amount=0.08):
        creative = profile.copy()
        for k in creative:
            delta = random.uniform(-amount, amount)
            creative[k] = max(0.01, min(creative[k] + delta, 0.99))
        return creative

    def corrected_profile(self, base_profile, track_features, strength=0.5):
        corrected = base_profile.copy()
        for k in base_profile:
            tf = track_features.get(k)
            if tf is None: continue
            corrected[k] = base_profile[k] + (base_profile[k] - tf) * strength
            corrected[k] = max(0.01, min(corrected[k], 0.99))
        return corrected

    def resolve_instruments(self, name):
        instr = INSTRUMENTS.get(name, instrument.Piano)
        return instr(), instrument.Piano()

    def resolve_lead_chord_instruments(self, lead_name, chord_name):
        lead_instr = INSTRUMENTS.get(lead_name, instrument.Piano)()
        chord_instr = INSTRUMENTS.get(chord_name, instrument.Piano)()
        return lead_instr, chord_instr

    def generate_track(self, profile, mood_override=None,
                       lead_instrument="Piano",
                       chord_instrument="Piano"):
        
        is_strict_mode = (mood_override is None)
        p = profile.copy()
        
        if not is_strict_mode:
            p.update(mood_override)
            def soft_clamp(value, target, strength=0.92, noise=0.0):
                val = target * strength + value * (1 - strength)
                val += random.uniform(-noise, noise)
                return max(0.01, min(val, 0.99))
            
            track_features = p.copy()
            for k in ['energy', 'depth', 'complexity', 'openness', 'structure']:
                track_features[k] = soft_clamp(track_features.get(k, 0.5), p[k], strength=0.85, noise=0.07)
            p = track_features

        bpm = int(60 + p['energy'] * 120)
        mode = 'minor' if p['depth'] > 0.55 else 'major'
        
        melody_instr, harmony_instr = self.resolve_lead_chord_instruments(lead_instrument, chord_instrument)
        
        root_notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        root = random.choice(root_notes)
        melody_key = key.Key(root, mode=mode)
        harmony_key = key.Key(root, mode=mode)

        scale_pitches = [pitch.midi for pitch in melody_key.getPitches("G3", "C6")]
        center_idx = len(scale_pitches) // 2
        range_width = int(len(scale_pitches) * (0.2 + 0.6 * p['openness']))
        min_idx = max(0, center_idx - range_width // 2)
        max_idx = min(len(scale_pitches) - 1, center_idx + range_width // 2)

        s = stream.Score()
        part_melody = stream.Part()
        part_harmony = stream.Part()
        part_drums = stream.Part()
        part_bass = stream.Part()
        
        part_melody.append(tempo.MetronomeMark(number=bpm))
        part_melody.insert(0, melody_instr)
        part_harmony.insert(0, harmony_instr)
        
        # drums on the 10ch
        perc_inst = instrument.Percussion()
        try: perc_inst.midiChannel = 9 
        except: pass
        part_drums.insert(0, perc_inst)
        
        #bass
        bass_inst = instrument.ElectricBass()
        part_bass.insert(0, bass_inst)
        
        part_melody.append(melody_key)
        part_harmony.append(harmony_key)

        base_prog = ['I', 'V', 'vi', 'IV'] if mode == 'major' else ['i', 'VI', 'III', 'VII']
        progression = base_prog * 8
        
        chord_objs = []
        for rom in progression:
            rn = roman.RomanNumeral(rom, harmony_key)
            c = chord.Chord(rn.pitches)
            c.quarterLength = 4.0
            c.closedPosition(forceOctave=3, inPlace=True)
            part_harmony.append(c)
            chord_objs.append(c)

        #melody
        curr_index = center_idx 
        best_dist = 999
        for i in range(min_idx, max_idx+1):
            if scale_pitches[i] % 12 == melody_key.tonic.midi % 12:
                if abs(i - center_idx) < best_dist:
                    best_dist = abs(i - center_idx)
                    curr_index = i

        total_beats = len(progression) * 4
        current_beat = 0
        rhythm_patterns = [[1.0, 1.0], [0.5, 1.0, 0.5], [2.0], [1.5, 0.5], [0.5, 0.5, 1.0], [1.0, 0.5, 0.5]]
        
        if is_strict_mode:
            if p['complexity'] < 0.25: allowed_steps = [1, -1, 1, -1, 2, -2] 
            elif p['complexity'] < 0.6: allowed_steps = [0, 1, -1, 1, -1, 2, -2, 3, -3]
            else: allowed_steps = [1, -1, 2, -2, 3, -3, 4, -4, 0]
        else:
            allowed_steps = [0, 1, -1, 1, -1, 2, -2, 3, -3]

        arp_patterns = [[0, 1, 2, 1], [0, 2, 1, 0], [0, 1, 2, 3]] 
        current_rhythm = random.choice(rhythm_patterns)
        
        while current_beat < total_beats:
            if is_strict_mode:
                if p['structure'] < 0.8 and random.random() < (1 - p['structure']):
                     current_rhythm = random.choice(rhythm_patterns)
            else:
                if random.random() < 0.4: current_rhythm = random.choice(rhythm_patterns)

            rhythm = current_rhythm
            use_arp = False
            if p['complexity'] > 0.3 or (not is_strict_mode and random.random() < 0.3):
                use_arp = random.random() < 0.3
            arp = random.choice(arp_patterns) if use_arp else None

            for i in range(len(rhythm)):
                dur = rhythm[i]
                if current_beat + dur > total_beats:
                    dur = total_beats - current_beat
                    if dur <= 0: break

                chord_idx = int(current_beat // 4)
                chord_obj = chord_objs[chord_idx] if chord_idx < len(chord_objs) else None
                chord_tones_midi = [pit.midi % 12 for pit in chord_obj.pitches] if chord_obj else []

                if chord_tones_midi and arp:
                    possible_indices = [idx for idx in range(min_idx, max_idx) if scale_pitches[idx] % 12 in chord_tones_midi]
                    if possible_indices:
                        closest_idx = min(possible_indices, key=lambda x: abs(x - curr_index))
                        arp_offset = arp[i % len(arp)]
                        try:
                            curr_pos_in_chord = 0
                            for k, p_idx in enumerate(possible_indices):
                                if p_idx == closest_idx: curr_pos_in_chord = k; break
                            next_index = possible_indices[(curr_pos_in_chord + arp_offset) % len(possible_indices)]
                        except: next_index = closest_idx
                    else: next_index = curr_index
                else:
                    step = random.choice(allowed_steps)
                    if is_strict_mode and p['complexity'] < 0.3 and step == 0: step = random.choice([-1, 1])
                    next_index = curr_index + step

                next_index = max(min_idx, min(max_idx, next_index))
                candidate_pitch = scale_pitches[next_index]
                
                should_harmonize = False
                if chord_tones_midi and (candidate_pitch % 12) not in chord_tones_midi:
                    if is_strict_mode or random.random() < p['structure']: should_harmonize = True
                
                if should_harmonize and chord_tones_midi:
                    search_offsets = [0, 1, -1, 2, -2]
                    for off in search_offsets:
                        check_idx = next_index + off
                        if min_idx <= check_idx <= max_idx:
                            if (scale_pitches[check_idx] % 12) in chord_tones_midi:
                                next_index = check_idx; break
                
                curr_index = next_index
                final_pitch = scale_pitches[curr_index]

                n = note.Note(final_pitch)
                n.quarterLength = dur
                n.volume.velocity = int(55 + p['energy'] * 55)
                part_melody.append(n)
                current_beat += dur
                if current_beat >= total_beats: break

        # bass sound
        for i in range(len(progression)):
            c_obj = chord_objs[i]
            root_midi = c_obj.root().midi
            while root_midi > 52: root_midi -= 12
            while root_midi < 28: root_midi += 12
            
            # bass sounds like dis
            if p['energy'] > 0.6:
                pattern = [(1.0, True), (2.0, False), (1.0, True)]
                if p['complexity'] > 0.6:
                    pattern = [(0.5, True), (0.5, True), (0.5, False), (0.5, True)] * 2
            else:
                pattern = [(4.0, True)]

            for dur, is_note in pattern:
                if is_note:
                    n = note.Note(root_midi)
                    n.quarterLength = dur
                    n.volume.velocity = 90
                    part_bass.append(n)
                else:
                    r = note.Rest()
                    r.quarterLength = dur
                    part_bass.append(r)

        # drums sound like dis
        drum_duration = 0.5 
        total_eighths = int(total_beats * 2)
        
        for i in range(total_eighths):
            beat_pos = i / 2.0 
            notes_to_play = []
            is_crash_hit = (beat_pos % 16 == 0)
            
            if is_crash_hit:
                notes_to_play.append(49)
                notes_to_play.append(36)
            else:
                notes_to_play.append(42)
            
            pos_in_bar = beat_pos % 4.0
            if not is_crash_hit:
                if pos_in_bar == 0.0 or pos_in_bar == 2.0: notes_to_play.append(36)
            if pos_in_bar == 1.0 or pos_in_bar == 3.0: notes_to_play.append(38)

            if notes_to_play:
                notes_to_play = list(set(notes_to_play))
                if len(notes_to_play) > 1:
                    c = chord.Chord(notes_to_play)
                    c.quarterLength = drum_duration
                    c.volume.velocity = 85
                    part_drums.append(c)
                else:
                    n = note.Note(notes_to_play[0])
                    n.quarterLength = drum_duration
                    n.volume.velocity = 85
                    part_drums.append(n)
            else:
                r = note.Rest()
                r.quarterLength = drum_duration
                part_drums.append(r)

        s.append(part_melody) 
        s.append(part_harmony) 
        s.append(part_drums)  
        s.append(part_bass) 
        return s

#gui carousel
class InstrumentCarousel(tk.Frame):
    def __init__(self, parent, title, options, variable, *args, **kwargs):
        super().__init__(parent, bg=COLORS["bg_panel"], *args, **kwargs)
        self.options = options
        self.variable = variable
        self.current_idx = 0
        
        start_val = variable.get()
        if start_val in options: self.current_idx = options.index(start_val)
            
        self.width = 120
        self.height = 100
        self.animation_running = False

        lbl = tk.Label(self, text=title, bg=COLORS["bg_panel"], fg=COLORS["text_dim"], font=("Arial", 10))
        lbl.pack(pady=(5, 0))

        container = tk.Frame(self, bg=COLORS["bg_panel"])
        container.pack(pady=5)

        self.btn_left = tk.Label(container, text="◀", bg=COLORS["bg_panel"], fg=COLORS["text_dim"], 
                                 font=("Arial", 18), cursor="hand2")
        self.btn_left.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        self.btn_left.bind("<Button-1>", lambda e: self.prev_item())
        self.btn_left.bind("<Enter>", lambda e: self.btn_left.config(fg=COLORS["text_main"]))
        self.btn_left.bind("<Leave>", lambda e: self.btn_left.config(fg=COLORS["text_dim"]))

        self.canvas = tk.Canvas(container, width=self.width, height=self.height, 
                                bg=COLORS["bg_main"], highlightthickness=0) 
        self.canvas.pack(side=tk.LEFT, padx=5)
  
        self.btn_right = tk.Label(container, text="▶", bg=COLORS["bg_panel"], fg=COLORS["text_dim"], 
                                  font=("Arial", 18), cursor="hand2")
        self.btn_right.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        self.btn_right.bind("<Button-1>", lambda e: self.next_item())
        self.btn_right.bind("<Enter>", lambda e: self.btn_right.config(fg=COLORS["text_main"]))
        self.btn_right.bind("<Leave>", lambda e: self.btn_right.config(fg=COLORS["text_dim"]))

        self.lbl_selected = tk.Label(self, text="Automatically chosen", bg=COLORS["bg_panel"], fg=COLORS["text_dim"], font=("Arial", 8))
        self.lbl_selected.pack(pady=(0,5))

        self.draw_item(self.current_idx, 0)
        self.update_variable()

    def get_icon(self, name): return INSTRUMENT_ICONS.get(name, "🎵")
    def draw_item(self, index, offset_x):
        self.canvas.delete("all")
        name = self.options[index]
        icon = self.get_icon(name)
        cx = self.width / 2 + offset_x
        cy = self.height / 2
        self.canvas.create_rectangle(2+offset_x, 2, self.width-2+offset_x, self.height-2, outline=COLORS["btn_green"], width=2)
        self.canvas.create_text(cx, cy - 10, text=icon, font=("Segoe UI Emoji", 40), fill="white")
        short_name = name.split()[0] if len(name) > 12 else name
        self.canvas.create_text(cx, cy + 35, text=short_name, font=("Arial", 10, "bold"), fill="white")
    def next_item(self):
        if self.animation_running: return
        self.animate_slide(1)
    def prev_item(self):
        if self.animation_running: return
        self.animate_slide(-1)
    def animate_slide(self, direction):
        self.animation_running = True
        steps = 8
        next_idx = (self.current_idx + direction) % len(self.options)
        w = self.width
        def step_anim(s):
            progress = s / steps
            offset_curr = -w * direction * progress
            offset_next = w * direction * (1 - progress)
            self.canvas.delete("all")
            self.draw_item(self.current_idx, offset_curr)
            self.draw_item(next_idx, offset_next)
            if s < steps: self.after(15, lambda: step_anim(s + 1))
            else:
                self.current_idx = next_idx
                self.draw_item(self.current_idx, 0)
                self.update_variable()
                self.animation_running = False
        step_anim(1)
    def update_variable(self):
        self.variable.set(self.options[self.current_idx])

class PianoKeyboard(tk.Canvas):
    def __init__(self, parent, width=700, height=100, start_note=48, end_note=84):
        # 48 = C3 / 84 = C6
        super().__init__(parent, width=width, height=height, bg="#333", highlightthickness=0)
        self.start_note = start_note
        self.end_note = end_note
        self.white_keys = []
        self.black_keys = []

        total_keys = 0
        for i in range(start_note, end_note + 1):
            if i % 12 in [0, 2, 4, 5, 7, 9, 11]: # white klavishi
                total_keys += 1
        
        self.key_w = width / total_keys
        self.key_h = height
        self.draw_keyboard()
        
        self.active_notes = set()

    def draw_keyboard(self):
        self.delete("all")
        self.key_map = {} # midi -> canvas_id
        
        x = 0
        # white keys
        for pitch in range(self.start_note, self.end_note + 1):
            note_in_octave = pitch % 12
            is_white = note_in_octave in [0, 2, 4, 5, 7, 9, 11]
            
            if is_white:
                rect = self.create_rectangle(x, 0, x + self.key_w, self.key_h, 
                                             fill="white", outline="black", tags=f"key_{pitch}")
                self.key_map[pitch] = rect
                # C note text
                if note_in_octave == 0:
                    self.create_text(x + self.key_w/2, self.key_h - 15, text=f"C{pitch//12 - 1}", fill="#ccc", font=("Arial", 8))
                x += self.key_w

        # black keys
        x = 0
        for pitch in range(self.start_note, self.end_note + 1):
            note_in_octave = pitch % 12
            is_white = note_in_octave in [0, 2, 4, 5, 7, 9, 11]
            
            if is_white:
                x += self.key_w
            else:
                #black a lil left
                bx = x - (self.key_w * 0.35)
                rect = self.create_rectangle(bx, 0, bx + self.key_w * 0.7, self.key_h * 0.6, 
                                             fill="black", outline="black", tags=f"key_{pitch}")
                self.key_map[pitch] = rect

    def set_active_keys(self, notes):
        #updatin
        for n in self.active_notes:
            if n not in notes and n in self.key_map:
                is_black = (n % 12) not in [0, 2, 4, 5, 7, 9, 11]
                color = "black" if is_black else "white"
                self.itemconfig(self.key_map[n], fill=color)

        for n in notes:
            if n in self.key_map:
                # pressing button color
                self.itemconfig(self.key_map[n], fill="#00ccff")
        
        self.active_notes = set(notes)

class OnboardingWindow(tk.Toplevel):
    def __init__(self, parent_app):
        super().__init__(parent_app.root, bg=COLORS["bg_main"])
        self.app = parent_app
        self.title("Tuning music taste")
        self.geometry("500x450")
        self.resizable(False, False)
        
        self.app.root.withdraw()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.total_steps = 10
        self.current_step = 0
        self.current_features = None
        
        self.setup_ui()
        self.after(500, self.play_next_track)

    def setup_ui(self):
        tk.Label(self, text="Welcome!", 
                 bg=COLORS["bg_main"], fg="white", font=("Arial", 18, "bold")).pack(pady=(30, 5))
        
        tk.Label(self, text="Listen to 10 short melodies.\nEvaluate them\nIt will help to create your music profile", 
                 bg=COLORS["bg_main"], fg=COLORS["text_dim"], font=("Arial", 11)).pack(pady=10)
        
        self.lbl_progress = tk.Label(self, text=f"Track 1 of {self.total_steps}", 
                                     bg=COLORS["bg_main"], fg=COLORS["highlight"], font=("Arial", 12, "bold"))
        self.lbl_progress.pack(pady=15)
        
        self.lbl_status = tk.Label(self, text="Preparing...", bg=COLORS["bg_main"], fg=COLORS["text_dim"])
        self.lbl_status.pack(pady=5)

        btn_frame = tk.Frame(self, bg=COLORS["bg_main"])
        btn_frame.pack(pady=30)

        self.btn_like = ShadowButton(btn_frame, text="👍 Like", fg_color=COLORS["text_success"],
                                     width=14, height=2, command=lambda: self.vote(1.0))
        self.btn_like.pack(side=tk.LEFT, padx=15)
        
        self.btn_dislike = ShadowButton(btn_frame, text="👎 Dislike", fg_color=COLORS["text_error"],
                                        width=14, height=2, command=lambda: self.vote(-1.0))
        self.btn_dislike.pack(side=tk.LEFT, padx=15)
        
        self.disable_buttons()

    def play_next_track(self):
        if self.current_step >= self.total_steps:
            self.finish_calibration()
            return
            
        self.current_step += 1
        self.lbl_progress.config(text=f"Track {self.current_step} of {self.total_steps}")
        self.lbl_status.config(text="Generating...", fg=COLORS["text_dim"])
        self.disable_buttons()
        self.update()

        random_profile = {k: random.random() for k in ['energy','depth','complexity','openness','structure']}
        
        try:
            score = self.app.generator.generate_track(random_profile, lead_instrument="Piano", chord_instrument="Piano")
            self.current_features = self.app.analyst.analyze_stream(score)
            
            test_score = stream.Score()
            test_score.append(score.parts[0]) # lead
            if len(score.parts) > 1: test_score.append(score.parts[1]) # chords
            if len(score.parts) > 3: test_score.append(score.parts[3]) # bass

            mm = score.recurse().getElementsByClass(tempo.MetronomeMark)
            if mm: test_score.insert(0, mm[0])
                
            test_midi_path = os.path.abspath("temp_onboarding.mid")
            test_score.write('midi', fp=test_midi_path)

            pygame.mixer.music.load(test_midi_path)
            pygame.mixer.music.play()
            self.lbl_status.config(text="Playing...", fg=COLORS["highlight"])
            self.enable_buttons()
            
        except Exception as e:
            print(f"Test error: {e}")
            self.vote(0) 

    def vote(self, rating):
        pygame.mixer.music.fadeout(300)
        if self.current_features:
            self.app.user.update_profile(self.current_features, rating)

        self.after(400, self.play_next_track)

    def finish_calibration(self):
        pygame.mixer.music.stop()
        self.app.update_chart() 
        self.app.root.deiconify() 
        self.destroy()

    def disable_buttons(self):
        self.btn_like.config_state("disabled")
        self.btn_dislike.config_state("disabled")
    def enable_buttons(self):
        self.btn_like.config_state("normal")
        self.btn_dislike.config_state("normal")
    def on_close(self):
        self.finish_calibration()

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Neuromusix")
        self.root.geometry("1100x900") 

        self.user = UserProfile()
        self.generator = MelodyGenerator()
        self.analyst = MusicAnalyst()

        self.current_score = None
        self.current_features = None
        self.midi_file = "temp_gen.mid"
        self.current_bpm = None
        self.note_timeline = [] 
        self.is_playing = False 
        
        self.drums_enabled = tk.BooleanVar(value=True)
        self.bass_enabled = tk.BooleanVar(value=True)

        try:
            pygame.init()
            pygame.mixer.init()
        except:
            print("Audio initialization error")

        self.setup_ui()

        if self.user.profile['energy'] == 0.5 and self.user.profile['complexity'] == 0.5:
            self.root.after(100, self.initial_user_test)

    def setup_ui(self):
        left_panel = tk.Frame(self.root, bg=COLORS["bg_panel"], width=340) 
        left_panel.pack(side=tk.LEFT, fill=tk.Y)
        left_panel.pack_propagate(False)

        right_panel = tk.Frame(self.root, bg=COLORS["bg_main"]) 
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = plt.Figure(figsize=(5,4), dpi=100) 
        self.fig.patch.set_facecolor(COLORS["bg_main"])
        self.ax = self.fig.add_subplot(111, polar=True)
        self.ax.set_facecolor(COLORS["graph_bg"]) 
        self.ax.spines['polar'].set_color(COLORS["text_dim"])
        self.ax.tick_params(axis='x', colors=COLORS["text_main"])
        self.ax.tick_params(axis='y', colors=COLORS["text_dim"])
        
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)
        self.update_chart()

        tk.Label(right_panel, text="Melody visualization", bg=COLORS["bg_main"], fg=COLORS["text_dim"], font=("Arial", 10)).pack(pady=(0,5))
        self.piano = PianoKeyboard(right_panel, height=80, start_note=45, end_note=96)
        self.piano.configure(bg=COLORS["bg_main"]) 
        self.piano.pack(padx=20, pady=20)

        try:
            original_img = Image.open("logo.png")
            target_width = 170
            aspect_ratio = original_img.height / original_img.width
            target_height = int(target_width * aspect_ratio)

            resized_img = original_img.resize((target_width, target_height), Image.Resampling.LANCZOS)

            self.logo_img = ImageTk.PhotoImage(resized_img)

            tk.Label(left_panel, image=self.logo_img, bg="#131313", bd=0).pack(pady=(25, 15))
            
        except Exception as e:
            print(f"Не удалось загрузить логотип: {e}")
            tk.Label(left_panel, text="neuromusix", font=("Arial", 28, "bold"), 
                     bg="#131313", fg="white").pack(pady=15)

        gen_frame = tk.Frame(left_panel, bg=COLORS["bg_panel"]) 
        gen_frame.pack(fill=tk.X, padx=15, pady=3)
  
        self.btn_gen = ShadowButton(gen_frame, text="Generate", 
                                  bg_color=COLORS["btn_green"], hover_color=COLORS["btn_green_hover"],
                                  width=16, height=2, command=self.generate_music)
        self.btn_gen.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
        
        self.btn_rand = ShadowButton(gen_frame, text="🎲 RANDOM", 
                                   bg_color=COLORS["btn_base"], hover_color=COLORS["btn_hover"],
                                   width=16, height=2, command=self.generate_random)
        self.btn_rand.pack(side=tk.LEFT, padx=(5, 0), fill=tk.X, expand=True)

        play_frame = tk.Frame(left_panel, bg=COLORS["bg_panel"]) 
        play_frame.pack(fill=tk.X, padx=15, pady=10)
        
        self.btn_play = ShadowButton(play_frame, text="Play ▶", 
                                   width=16, height=2, command=self.play_music)
        self.btn_play.config_state("disabled")
        self.btn_play.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
        
        self.btn_stop = ShadowButton(play_frame, text="Stop ■", 
                                   width=16, height=2, command=self.stop_music)
        self.btn_stop.config_state("disabled")
        self.btn_stop.pack(side=tk.LEFT, padx=(5, 0), fill=tk.X, expand=True)

        bpm_frame = tk.Frame(left_panel, bg=COLORS["bg_panel"]) 
        bpm_frame.pack(fill=tk.X, padx=15, pady=5)
        
        tk.Label(bpm_frame, text="BPM:", bg=COLORS["bg_panel"], fg=COLORS["text_dim"]).pack(side=tk.LEFT)
        self.bpm_var = tk.StringVar()
        self.bpm_entry = tk.Entry(bpm_frame, textvariable=self.bpm_var, justify="center", width=5, 
                                  bg="#333", fg="white", insertbackground="white")
        self.bpm_entry.pack(side=tk.LEFT, padx=5)
        
        ShadowButton(bpm_frame, text="Apply", width=8, height=1, font=("Arial", 9),
                   command=self.apply_bpm).pack(side=tk.RIGHT)

        tk.Label(left_panel, text="Mood:", bg=COLORS["bg_panel"], fg=COLORS["text_dim"]).pack(pady=(10,2))
        
        self.override_var = tk.StringVar(value="By my profile")
        
        mood_frame = tk.Frame(left_panel, bg=COLORS["bg_panel"])
        mood_frame.pack(fill=tk.X, padx=15)
        
        self.btn_mood = ShadowButton(mood_frame, text="By my profile ▼", 
                                     bg_color=COLORS["btn_base"], hover_color=COLORS["btn_hover"],
                                     width=24, height=1, font=("Arial", 10))
        self.btn_mood.pack()
        
        self.mood_menu = tk.Menu(self.root, tearoff=0, 
                                 bg=COLORS["btn_base"], fg=COLORS["text_main"], 
                                 activebackground=COLORS["highlight"], activeforeground="white",
                                 relief="flat", bd=0)
        
        modes = ["By my profile", "Energetic", "Sad/deep", "Complex/jazz", "Calm/atmospheric", "Light/dreamy"]
        
        def set_mood(val):
            self.override_var.set(val)
            self.btn_mood.lbl.config(text=f"{val} ▼") 
            
        for m in modes:
            self.mood_menu.add_command(label=m, command=lambda val=m: set_mood(val))


        def show_mood_menu():
            x = self.btn_mood.lbl.winfo_rootx()
            y = self.btn_mood.lbl.winfo_rooty() + self.btn_mood.lbl.winfo_height()
            self.mood_menu.post(x, y)
            
        self.btn_mood.command = show_mood_menu

        chk_frame = tk.Frame(left_panel, bg=COLORS["bg_panel"])
        chk_frame.pack(fill=tk.X, padx=15, pady=(15,0))
        
        self.chk_drums = tk.Checkbutton(chk_frame, text="🥁 Drums", variable=self.drums_enabled,
            bg=COLORS["bg_panel"], fg=COLORS["text_main"], selectcolor=COLORS["bg_panel"], 
            activebackground=COLORS["bg_panel"], activeforeground="white")
        self.chk_drums.pack(side=tk.LEFT)
        
        self.chk_bass = tk.Checkbutton(chk_frame, text="🎸 Bass", variable=self.bass_enabled,
            bg=COLORS["bg_panel"], fg=COLORS["text_main"], selectcolor=COLORS["bg_panel"],
            activebackground=COLORS["bg_panel"], activeforeground="white")
        self.chk_bass.pack(side=tk.RIGHT)

        instr_names = list(INSTRUMENTS.keys())
        self.instrument_var = tk.StringVar(value="Piano")
        self.carousel_lead = InstrumentCarousel(left_panel, "Instrument (Lead)", instr_names, self.instrument_var)
        self.carousel_lead.pack(pady=5)
        self.chord_instrument_var = tk.StringVar(value="Piano")
        self.carousel_chord = InstrumentCarousel(left_panel, "Instrument (Chords)", instr_names, self.chord_instrument_var)
        self.carousel_chord.pack(pady=5)

        rate_frame = tk.Frame(left_panel, bg=COLORS["bg_panel"])
        rate_frame.pack(fill=tk.X, padx=15, pady=(15,5))
        
        self.btn_like = ShadowButton(rate_frame, text="👍 Like", fg_color="#2ECC71",
                                   width=16, height=2, command=lambda: self.give_feedback(1.0))
        self.btn_like.config_state("disabled")
        self.btn_like.pack(side=tk.LEFT, padx=(0,2), fill=tk.X, expand=True)
        
        self.btn_dislike = ShadowButton(rate_frame, text="👎 Dislike", fg_color="#E74C3C",
                                      width=16, height=2, command=lambda: self.give_feedback(-1.0))
        self.btn_dislike.config_state("disabled")
        self.btn_dislike.pack(side=tk.LEFT, padx=(2,0), fill=tk.X, expand=True)

        footer_frame = tk.Frame(left_panel, bg=COLORS["bg_panel"])
        footer_frame.pack(fill=tk.X, padx=15, pady=10)
        
        ShadowButton(footer_frame, text="Reset profile", 
                     bg_color=COLORS["btn_red"], hover_color=COLORS["btn_red_hover"],
                     width=12, height=2, font=("Arial", 9, "bold"),
                     command=self.reset_profile).pack(side=tk.LEFT, padx=(0,5), fill=tk.X, expand=True)

        ShadowButton(footer_frame, text="Export 💾", 
                     bg_color=COLORS["btn_blue"], hover_color=COLORS["btn_blue_hover"],
                     width=12, height=2, font=("Arial", 9, "bold"),
                     command=self.open_export_window).pack(side=tk.LEFT, padx=(5,0), fill=tk.X, expand=True)

        self.btn_instruction = ShadowButton(left_panel, text="📄 Instruction", 
                                            bg_color=COLORS["btn_base"], 
                                            hover_color=COLORS["btn_hover"],
                                            fg_color=COLORS["text_dim"],
                                            width=30, height=1,
                                            font=("Arial", 10),
                                            command=self.open_manual)
        self.btn_instruction.pack(side=tk.BOTTOM, padx=20, pady=(10, 20))

        self.lbl_status = tk.Label(left_panel, text="Ready", bg=COLORS["bg_panel"], fg=COLORS["btn_green"])
        self.lbl_status.pack(side=tk.BOTTOM, pady=2)

    def open_export_window(self):
        if not self.current_score:
            messagebox.showwarning("Warning", "You need to generate a track first!")
            return
        
        win = tk.Toplevel(self.root)
        win.title("Export")
        win.geometry("300x250")
        
        tk.Label(win, text="Choose the format:", font=("Arial", 12, "bold")).pack(pady=10)
        tk.Button(win, text="MIDI (.mid)", width=20, command=lambda: self.export_midi(win)).pack(pady=5)
        tk.Button(win, text="Audio (.wav)", width=20, command=lambda: self.export_wav(win)).pack(pady=5)
        tk.Button(win, text="MuseScore (.mxl)", width=20, command=lambda: self.export_musescore(win)).pack(pady=5)
        tk.Button(win, text="Cancel", command=win.destroy).pack(pady=15)
    
    def export_midi(self, window):
        path = filedialog.asksaveasfilename(defaultextension=".mid", filetypes=[("MIDI files", "*.mid")])
        if path:
            self.save_current_selection(path)
            messagebox.showinfo("Success", f"Saved: {path}")
            window.destroy()
    def export_wav(self, window):
        #errors
        sf_abspath = os.path.abspath(SOUNDFONT_PATH)
        
        if not os.path.exists(sf_abspath):
             messagebox.showerror("SoundFont Error", f"Not found (.sf2)!\n\nSystem finds soundfont.sf2 there:\n{sf_abspath}\n\n")
             return

        #fluidsynth
        from shutil import which
        if which("fluidsynth") is None:
             messagebox.showerror("Error", "FluidSynth is not installed or could not be found in the path.")
             return

        # choose method of saving
        wav_abspath = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
        if not wav_abspath:
            return
        
        # closing window stoppin player
        window.destroy()
        self.stop_music()
        self.root.update()

        # preparing a midi file
        temp_mid = os.path.abspath("temp_export.mid")
        try:
            self.save_current_selection(temp_mid)
        except Exception as e:
            messagebox.showerror("MIDI error", f"Failed to create temporary file: {e}")
            return

        # export to wav
        self.lbl_status.config(text="Exporting to WAV... (Don't close)")
        
        def run_conversion_thread():
            try:
                cmd = [
                    "fluidsynth",
                    "-ni",                  # no interface
                    "-F", wav_abspath,      # render mode
                    "-r", "44100",          # khz
                    sf_abspath,             # soundfont
                    temp_mid                # midi file
                ]
                process = subprocess.run(cmd, capture_output=True, text=True)
                
                #chheckin the result
                if process.returncode == 0 and os.path.exists(wav_abspath) and os.path.getsize(wav_abspath) > 0:
                    self.root.after(0, lambda: messagebox.showinfo("Success", f"File saved:\n{wav_abspath}"))
                    self.root.after(0, lambda: self.lbl_status.config(text="Successful export"))
                else:
                    # error fluidsynth
                    err_log = process.stderr if process.stderr else process.stdout
                    self.root.after(0, lambda: messagebox.showerror("FluidSynth export", f"WAV file couldn't be saved\n\nЛог:\n{err_log}"))
                    self.root.after(0, lambda: self.lbl_status.config(text="Export error"))

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Critical error", str(e)))
            
            finally:
                if os.path.exists(temp_mid):
                    try:
                        os.remove(temp_mid)
                    except: pass

        # thread starts
        t = threading.Thread(target=run_conversion_thread, daemon=True)
        t.start()

    def export_musescore(self, window):
        try:
            xml_filename = "composition.musicxml"
            self.current_score.write('musicxml', fp=xml_filename)
            musescore_path = '/Applications/MuseScore 4.app/Contents/MacOS/mscore'
            subprocess.Popen([musescore_path, xml_filename])
            window.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"MuseScore 4 could not be opened: {e}")

    def save_current_selection(self, filepath):
        save_score = stream.Score()
        save_score.append(self.current_score.parts[0]) # melody
        save_score.append(self.current_score.parts[1]) # harmony
        
        if self.drums_enabled.get() and len(self.current_score.parts) > 2:
            save_score.append(self.current_score.parts[2]) #drums
            
        if self.bass_enabled.get() and len(self.current_score.parts) > 3:
             save_score.append(self.current_score.parts[3]) #basssss
        save_score.write('midi', fp=filepath)

    def open_manual(self):
        filename = "manual.txt"
        try:
            if platform.system() == 'Darwin': subprocess.call(('open', filename))
            elif platform.system() == 'Windows': os.startfile(filename)
            else: subprocess.call(('xdg-open', filename))
        except: pass

#other methods
    def initial_user_test(self):
        OnboardingWindow(self)

    def update_chart(self):
        self.ax.clear()
        categories = ['energy','depth','complexity','openness','structure']
        values = self.user.get_vector()
        values += values[:1]
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        self.ax.set_theta_offset(np.pi/2)
        self.ax.set_theta_direction(-1)
        self.ax.set_rlabel_position(0)
        self.ax.set_ylim(0,1)
        self.ax.set_xticks(angles[:-1])
        self.ax.set_xticklabels(categories)
        self.ax.plot(angles, values, linewidth=2, linestyle='solid', label="Your profile")
        self.ax.fill(angles, values, 'b', alpha=0.1)
        if self.current_features:
            track_vals = [self.current_features.get(k.lower(), 0.5) for k in categories]
            track_vals += track_vals[:1]
            self.ax.plot(angles, track_vals, linewidth=1, linestyle='dashed', color='red', label="Track")
        self.ax.legend(loc='upper right', bbox_to_anchor=(0.1,0.1))
        self.canvas_plot.draw()

    def generate_random(self):
        random_profile = {k: min(0.99, max(0.01, v + random.uniform(-0.25, 0.25))) for k,v in self.user.profile.items()}
        self._run_generation(random_profile, None)
        self.lbl_status.config(text="Random track generated")

    def generate_music(self):
        mode = self.override_var.get()
        override = None
        if mode == "Energetic": override = {'energy': 0.9, 'structure':0.8}
        elif mode == "Sad/deep": override = {'energy':0.3,'depth':0.9}
        elif mode == "Complex/jazz": override = {'complexity':0.9,'structure':0.2,'depth':0.6}
        elif mode == "Calm/atmospheric": override = {'energy': 0.25, 'openness': 0.7, 'structure': 0.6}
        elif mode == "Light/dreamy": override = {'energy': 0.5, 'depth': 0.3, 'openness': 0.8}

        if mode == "By my profile":
            self._run_generation(self.user.profile.copy(), None)
        else:
            p = self.generator.inject_creativity(self.user.profile, amount=0.1)
            self._run_generation(p, override)
        self.lbl_status.config(text="Successful generation")

    def _run_generation(self, profile, override):
        self.lbl_status.config(text="Generating...")
        self.root.update()
        self.current_score = self.generator.generate_track(
            profile, override,
            lead_instrument=self.instrument_var.get(),
            chord_instrument=self.chord_instrument_var.get()
        )
        self.current_features = self.analyst.analyze_stream(self.current_score)
        
        mm = self.current_score.recurse().getElementsByClass(tempo.MetronomeMark)
        if mm:
            self.current_bpm = int(mm[0].number)
            self.bpm_var.set(str(self.current_bpm))
        
        # piano building
        self.build_timeline() 
        
        self.update_current_instrument()
        self.btn_play.config_state("normal")
        self.btn_like.config_state("normal")
        self.btn_dislike.config_state("normal")
        self.update_chart()
        self.play_music()

    def apply_bpm(self):
        if not self.current_score: return
        try:
            new_bpm = int(self.bpm_var.get())
            if new_bpm < 30 or new_bpm > 300: raise ValueError
            
            self.current_bpm = new_bpm
            
            # updating
            for el in list(self.current_score.recurse()):
                if isinstance(el, tempo.MetronomeMark): 
                    self.current_score.remove(el, recurse=True)
            self.current_score.parts[0].insert(0, tempo.MetronomeMark(number=new_bpm))
            self.build_timeline() 

            self.update_current_instrument() #saves new midi
            self.lbl_status.config(text=f"BPM applied: {new_bpm}")
            
        except: messagebox.showerror("Error", "Incorrect BPM")
    def play_music(self):    
        self.update_current_instrument()
        try:
            # reloading thee file to reset the position
            pygame.mixer.music.load(self.midi_file)
            pygame.mixer.music.play(loops=-1)
        
            self.btn_play.config_state("disabled")
            self.btn_stop.config_state("normal")
            
            # piano animation
            self.is_playing = True
            self.animate_piano()
            
        except Exception as e:
            print(f"Playback error: {e}")

    def stop_music(self):
        pygame.mixer.music.stop()
        self.is_playing = False # stop the animation
        self.piano.set_active_keys([]) # notes stop!
        
        self.btn_play.config_state("normal")
        self.btn_stop.config_state("disabled")

    def animate_piano(self):
        if not self.is_playing:
            return
        current_time = pygame.mixer.music.get_pos()
        if current_time < 0: 
            # music not playing
            self.root.after(50, self.animate_piano)
            return
        
        active_pitches = [] #notes to sound
        
        for start, end, pitch in self.note_timeline:
            # small -50ms so the keys "stick"
            if start <= current_time < end - 10:
                active_pitches.append(pitch)
        
        self.piano.set_active_keys(active_pitches)
        
        self.root.after(30, self.animate_piano) # after 30ms repeating

    def update_current_instrument(self):
        if not self.current_score: return
        melody_instr, harmony_instr = self.generator.resolve_lead_chord_instruments(
            self.instrument_var.get(), self.chord_instrument_var.get())

        p0 = self.current_score.parts[0]
        for el in list(p0): 
            if isinstance(el, instrument.Instrument): p0.remove(el)
        p0.insert(0, melody_instr)

        if len(self.current_score.parts) > 1:
            p1 = self.current_score.parts[1]
            for el in list(p1): 
                if isinstance(el, instrument.Instrument): p1.remove(el)
            p1.insert(0, harmony_instr)

        self.save_current_selection(self.midi_file)

    def give_feedback(self, rating):
        if self.current_features:
            self.user.update_profile(self.current_features, rating)
            self.lbl_status.config(text="Profile updated")
            self.update_chart()
            self.btn_like.config_state("disabled")
            self.btn_dislike.config_state("disabled")
            
    def reset_profile(self):
        self.user.profile = {'energy': 0.5, 'depth': 0.5, 'complexity': 0.5, 'openness': 0.5, 'structure': 0.5}
        self.user.save_profile()
        self.update_chart()
        self.lbl_status.config(text="Profile reset")

    def open_in_musescore(self):
        if not self.current_score: return
        try:
            xml_filename = "composition.musicxml"
            self.current_score.write('musicxml', fp=xml_filename)
            subprocess.Popen(['/Applications/MuseScore 4.app/Contents/MacOS/mscore', xml_filename])
        except Exception as e: messagebox.showerror("Error", f"{e}")
    def build_timeline(self):
        self.note_timeline = []
        if not self.current_score or not self.current_bpm:
            return

        ms_per_quarter = 60000 / self.current_bpm
        
        parts_to_show = []
        if len(self.current_score.parts) > 0: parts_to_show.append(self.current_score.parts[0])
        if len(self.current_score.parts) > 1: parts_to_show.append(self.current_score.parts[1])

        for part in parts_to_show:
            for el in part.flat.notes:
                start_ms = el.offset * ms_per_quarter
                dur_ms = el.quarterLength * ms_per_quarter
                end_ms = start_ms + dur_ms
                
                if el.isChord:
                    for p in el.pitches:
                        self.note_timeline.append((start_ms, end_ms, p.midi))
                elif el.isNote:
                    self.note_timeline.append((start_ms, end_ms, el.pitch.midi))
        
        self.note_timeline.sort(key=lambda x: x[0])
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()