import { BorderBox } from "../atoms/BorderBox";

interface Props {
  checked: boolean;
  setChecked: React.Dispatch<React.SetStateAction<boolean>>;
}

export function ToggleSwitch({ checked, setChecked }: Props) {
  return (
    <BorderBox
      roundedFull
      className="flex h-8 w-14 cursor-pointer items-center p-1.5"
      onClick={() => setChecked((x) => !x)}
    >
      <input
        type="checkbox"
        role="switch"
        checked={checked}
        readOnly
        className="hidden"
      />
      <span
        className={
          "h-5 w-5 rounded-full border bg-white transition-all ease-in-out" +
          (checked ? ` translate-x-full` : " opacity-50")
        }
      />
    </BorderBox>
  );
}
