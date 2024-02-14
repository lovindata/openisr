import { BorderBox } from "../atoms/BorderBox";
import { SvgIcon } from "../atoms/SvgIcon";

interface Props {
  label: string;
  isLoading?: boolean;
  onClick?: React.MouseEventHandler<HTMLButtonElement>;
}

export function Button({ label, isLoading, onClick }: Props) {
  return (
    <button onClick={onClick} disabled={isLoading}>
      <BorderBox className="flex h-8 select-none items-center justify-center px-3 text-xs">
        {isLoading && (
          <SvgIcon type="loading" className="absolute h-4 w-4 animate-spin" />
        )}
        <label className={isLoading ? "opacity-0" : "cursor-pointer"}>
          {label}
        </label>
      </BorderBox>
    </button>
  );
}
