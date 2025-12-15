import os

PATH = "datasets/Human"

# Delete all files whose names start with "DEBUG" under PATH (recursive)
PREFIX = "DEBUG"

if not os.path.exists(PATH):
	print(f"Path '{PATH}' does not exist. Nothing to do.")
	raise SystemExit(0)

deleted = 0
errors = []
for root, dirs, files in os.walk(PATH):
	for fname in files:
		if fname.startswith(PREFIX):
			full = os.path.join(root, fname)
			try:
				os.remove(full)
				print(f"Deleted: {full}")
				deleted += 1
			except Exception as e:
				print(f"Failed to delete {full}: {e}")
				errors.append((full, str(e)))

print(f"Done. Deleted {deleted} files.")
if errors:
	print("Some deletions failed:")
	for p, e in errors:
		print(" -", p, e)
