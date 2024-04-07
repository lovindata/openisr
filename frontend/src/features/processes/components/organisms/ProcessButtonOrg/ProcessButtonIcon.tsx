import { SvgIconAtm } from "@/features/shared/components/atoms/SvgIconAtm";

interface Props {
  type: "download" | "error" | "run" | "stop";
  duration?: number;
  onClick?: () => void;
}

export function ProcessButtonIcon({ type, duration, onClick }: Props) {
  return (
    <div className="relative flex flex-col items-center">
      <SvgIconAtm
        type={type}
        className="h-6 w-6 cursor-pointer"
        onClick={onClick}
      />
      {duration && <span className="absolute top-full">{duration}s</span>}
    </div>
  );
}
