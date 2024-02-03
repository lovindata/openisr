import { BorderBox } from "../atoms/BorderBox";

interface Props {
  value: number | null;
  setValue: (_: number | null) => void;
  min?: number;
  max?: number;
  className?: string;
}

export function InputNumber({ value, setValue, min, max, className }: Props) {
  return (
    <BorderBox
      className={
        "flex h-8 items-center justify-center" +
        (className ? ` ${className}` : "")
      }
    >
      <input
        type="number"
        value={value?.toString() ?? ""}
        onChange={(event) => {
          let onChangeValue = parseInt(event.target.value);
          if (isNaN(onChangeValue)) {
            setValue(null);
            return;
          }
          onChangeValue = Math.max(min ?? onChangeValue, onChangeValue);
          onChangeValue = Math.min(max ?? onChangeValue, onChangeValue);
          setValue(onChangeValue);
        }}
        className="hide-spinner w-full bg-transparent text-center outline-none"
        min={min}
        max={max}
      />
    </BorderBox>
  );
}
