import { components } from "../../../../../services/backend/endpoints";
import { SvgIcon } from "../../../../atoms/SvgIcon";

interface Props {
  type: "download" | "error" | "run" | "stop";
  latestProcess?: components["schemas"]["ProcessODto"];
  onClick?: () => void;
}

export function Icon({ type, latestProcess, onClick }: Props) {
  return (
    <div className="relative flex flex-col items-center">
      <SvgIcon
        type={type}
        className="h-6 w-6 cursor-pointer"
        onClick={onClick}
      />
      {latestProcess && (
        <span className="absolute top-full">{`${latestProcess.status.duration}s`}</span>
      )}
    </div>
  );
}
