import { BorderBoxAtm } from "@/v2/features/shared/components/atoms/BorderBoxAtm";
import { SvgIconAtm } from "@/v2/features/shared/components/atoms/SvgIconAtm";

interface Props {
  value: string;
  onChange?: React.ChangeEventHandler<HTMLInputElement>;
}

export function SearchBarMol({ value, onChange }: Props) {
  return (
    <BorderBoxAtm className="flex h-8 w-72 items-center space-x-1 p-1">
      <SvgIconAtm type="search" className="h-6 w-6" />
      <input
        type="text"
        value={value}
        onChange={onChange}
        className="w-full bg-transparent outline-none"
      />
    </BorderBoxAtm>
  );
}
