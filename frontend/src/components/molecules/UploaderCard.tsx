import Folder from "../../assets/svgs/folder.svg?react";

export function UploaderCard() {
  return (
    <div className="relative h-16 w-32 rounded-lg border border-dashed">
      <input
        type="file"
        accept="image/*"
        multiple
        className="absolute inset-0 cursor-pointer opacity-0"
      />
      <div className="m-auto flex h-full w-min items-center space-x-3">
        <Folder className="h-8 min-h-max w-8 min-w-max fill-white" />
        <label className="">Local upload</label>
      </div>
    </div>
  );
}
