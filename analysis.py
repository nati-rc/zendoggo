from audio_utils import calculate_rms, calculate_pitch
import librosa
import numpy as np
from collections import defaultdict


def process_scores(scores, class_names, label_groups, special_labels):
    """
    Process model scores to group them by label_groups and handle special_labels. Special labels have a unique logic
    based ang get added to a group if either Dog, Cat or Bird sounds have been detected.
    """
    group_scores = defaultdict(float)
    for idx, score in enumerate(scores[0]):
        label = class_names[idx]
        if label in special_labels:
            continue  # Skip special labels in the initial pass
        group_name = next((g for g in label_groups if label in label_groups[g]), "Other")
        group_scores[group_name] = max(group_scores[group_name], score)

    # Handle special labels after the primary grouping
    for idx, score in enumerate(scores[0]):
        label = class_names[idx]
        if label not in special_labels:
            continue
        max_score = -1
        max_group = None
        for group in ["Dog", "Cat", "Bird"]:
            if group_scores[group] > max_score:
                max_score = group_scores[group]
                max_group = group
        if max_group:
            group_scores[max_group] = max(group_scores[max_group], score)
        else:
            group_scores["Other"] = max(group_scores["Other"], score)

    return group_scores

def analyze_segments(audio, intervals, sampling_rate, class_names, label_groups, special_labels, model, target_length, min_rms_threshold, min_pitch_prob_threshold):

    results = {'segments': [], 'total_audio_length_seconds': len(audio) / sampling_rate}

    for i, (start_idx, end_idx) in enumerate(intervals):
        segment_audio = audio[start_idx:end_idx]
        segment_audio = librosa.util.normalize(segment_audio)
        segment_audio = np.pad(segment_audio, (0, max(0, target_length - len(segment_audio))), mode='constant')
        segment_audio = segment_audio.flatten()

        rms = calculate_rms(segment_audio)
        _, pitch_prob = calculate_pitch(segment_audio, sampling_rate)

        if rms < min_rms_threshold or pitch_prob < min_pitch_prob_threshold:
            continue

        scores, embeddings, spectrogram = model(segment_audio)
        scores = scores.numpy()
        start_time, end_time = librosa.samples_to_time([start_idx, end_idx], sr=sampling_rate)

        group_scores = process_scores(scores, class_names, label_groups, special_labels)

        filtered_scores = {group: score for group, score in group_scores.items() if score >= 0.10}
        if filtered_scores:
            highest_category = max(filtered_scores, key=filtered_scores.get)
            #highest_score = filtered_scores[highest_category]
            results['segments'].append({'start_time': start_time, 'end_time': end_time, 'category': highest_category}) # add 'score': highest_score for debugging if needed

    return results
