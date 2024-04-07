import { BorderBoxAtm } from "@/features/shared/components/atoms/BorderBoxAtm";
import { SvgIconAtm } from "@/features/shared/components/atoms/SvgIconAtm";

interface Props {
  label: string;
  isLoading?: boolean;
  onClick?: React.MouseEventHandler<HTMLButtonElement>;
}

export function ButtonMol({ label, isLoading, onClick }: Props) {
  return (
    <button onClick={onClick} disabled={isLoading}>
      <BorderBoxAtm className="flex h-8 select-none items-center justify-center px-3 text-xs">
        {isLoading && (
          <SvgIconAtm
            type="loading"
            className="absolute h-4 w-4 animate-spin"
          />
        )}
        <label className={isLoading ? "opacity-0" : "cursor-pointer"}>
          {label}
        </label>
      </BorderBoxAtm>
    </button>
  );
}
