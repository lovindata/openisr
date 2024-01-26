import { BorderBox } from "../../atoms/BorderBox";
import { SvgIcon } from "../../atoms/SvgIcon";

interface Props {
  value: string;
  onChange?: React.ChangeEventHandler<HTMLInputElement>;
}

export function SearchBar({ value, onChange }: Props) {
  return (
    <BorderBox className="flex h-8 w-72 items-center space-x-1 p-1">
      <SvgIcon type="search" className="h-6 w-6" />
      <input
        type="text"
        value={value}
        onChange={onChange}
        className="w-full bg-transparent outline-none"
      />
    </BorderBox>
  );
}
