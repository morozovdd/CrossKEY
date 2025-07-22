from moviepy import VideoFileClip, concatenate_videoclips, vfx, clips_array

# Load GIFs
gif1 = VideoFileClip("assets/mr_us_volume_matches_x.gif")
gif2 = VideoFileClip("assets/mr_us_fov_comparison.gif")
gif3 = VideoFileClip("assets/registration_1.gif")
gif4 = VideoFileClip("assets/registration_2.gif")

# Get the maximum duration
max_duration = max(gif1.duration, gif2.duration, gif3.duration, gif4.duration)

# Function to loop or slow down clip to match target duration
def match_duration(clip, target_duration):
    if clip.duration < target_duration:
        # Repeat the clip as needed, then trim to exact duration
        loop_count = int(target_duration // clip.duration) + 1
        clip = concatenate_videoclips([clip] * loop_count).subclipped(0, target_duration)
    elif clip.duration > target_duration:
        # Slow down the clip (optional alternative to looping)
        clip = clip.fx(vfx.speedx, factor=clip.duration / target_duration)
    return clip

# Match durations
gif1 = match_duration(gif1, max_duration)
gif2 = match_duration(gif2, max_duration)
gif3 = match_duration(gif3, max_duration)
gif4 = match_duration(gif4, max_duration)

# Resize to uniform height
gif1 = gif1.resized(height=gif1.size[1])
gif2 = gif2.resized(height=gif1.size[1])
gif3 = gif3.resized(height=gif1.size[1])
gif4 = gif4.resized(height=gif1.size[1])

# Combine into a figure (horizontal layout, for example)
final_clip = clips_array([[gif1, gif3], [gif2, gif4]])

# Export as GIF
final_clip.write_gif("homogenized_combined.gif", fps=10)