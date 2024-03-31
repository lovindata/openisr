import { SvgIcon } from "@/v2/features/shared/components/atoms/SvgIcon";

interface Props {
  type: "download" | "error" | "run" | "stop";
  duration?: number;
  onClick?: () => void;
}

export function Icon({ type, duration, onClick }: Props) {
  return (
    <div className="relative flex flex-col items-center">
      <SvgIcon
        type={type}
        className="h-6 w-6 cursor-pointer"
        onClick={onClick}
      />
      {duration && <span className="absolute top-full">{duration}s</span>}
    </div>
  );
}
