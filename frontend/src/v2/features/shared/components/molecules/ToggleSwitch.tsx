import { BorderBox } from "@/v2/features/shared/components/atoms/BorderBox";

interface Props {
  checked: boolean;
  onSwitch: (_: boolean) => void;
}

export function ToggleSwitch({ checked, onSwitch: setChecked }: Props) {
  return (
    <BorderBox
      roundedFull
      className="flex h-8 w-14 cursor-pointer items-center p-1.5"
      onClick={() => setChecked(!checked)}
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
