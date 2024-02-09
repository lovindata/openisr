import { BorderBox } from "../atoms/BorderBox";

interface Props {
  label: string;
  onClick?: React.MouseEventHandler<HTMLButtonElement>;
  className?: string;
}

export function Button({ label, onClick, className }: Props) {
  return (
    <button onClick={onClick} className={className}>
      <BorderBox className="flex h-8 select-none items-center justify-center px-3">
        {label}
      </BorderBox>
    </button>
  );
}
