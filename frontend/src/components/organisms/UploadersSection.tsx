import { UploaderCard } from "../molecules/UploaderCard";

export function UploadersSection() {
  return (
    <div className="w-72 space-y-3">
      <h2 className="text-lg font-bold">Uploaders</h2>
      <div className="flex space-x-3">
        <UploaderCard />
        <UploaderCard />
      </div>
    </div>
  );
}
