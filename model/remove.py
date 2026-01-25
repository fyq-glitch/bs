import os
import shutil
def delete_file_recurisively(folder_path):
    if not os.path.isdir(folder_path):
        print(f"error,{folder_path} is not a directory")
        return
    files_to_delete=[]
    deleted_count=0
    failed_count=0
    print(f"Scanning {folder_path}")
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.lower().endswith((".npy")):
                file_path=os.path.join(dirpath, filename)
                files_to_delete.append(file_path)
    if not files_to_delete:
        print(f"no files found in {folder_path}")
        return
    print(f"deleting {len(files_to_delete)} files")
    confirm=input("Are you sure you want to delete these files? (y/n) ")
    if confirm == "y":
        print("Starting deleting")
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                deleted_count+=1
            except OSError as e:
                print(f"failed,{file_path}-error:{e}")
                failed_count+=1
        print(f"successfully deleted,{deleted_count},failed,{failed_count}")
    else:
        print("nothing to delete")
if __name__=="__main__":
    target_folder_path=r"C:\Users\fyq\Desktop\dataset"
    normalized_folder_path=os.path.normpath(target_folder_path)
    delete_file_recurisively(normalized_folder_path)
