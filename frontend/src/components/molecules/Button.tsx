import { BorderBox } from "../atoms/BorderBox";

interface Props {
  label: string;
  onClick?: React.MouseEventHandler<HTMLButtonElement>;
}

export function Button({ label, onClick }: Props) {
  return (
    <button onClick={onClick}>
      <BorderBox className="flex h-8 select-none items-center justify-center px-3 text-xs">
        {label}
      </BorderBox>
    </button>
  );
}
