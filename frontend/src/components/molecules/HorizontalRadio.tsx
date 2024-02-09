import { BorderBox } from "../atoms/BorderBox";
import { RadioGroup } from "@headlessui/react";

interface Props<T extends string> {
  possibleValues: T[];
  value: T;
  setValue: (_: T) => void;
  className?: string;
}

export function HorizontalRadio<T extends string>({
  possibleValues,
  value,
  setValue,
  className,
}: Props<T>) {
  return (
    <RadioGroup value={value} onChange={setValue}>
      <BorderBox
        className={
          "grid h-8 grid-flow-col justify-stretch divide-x text-xs" +
          (className ? ` ${className}` : "")
        }
      >
        {possibleValues.map((option, idx) => (
          <RadioGroup.Option value={option} key={idx} className="outline-none">
            {({ checked }) => (
              <label
                className={
                  "flex h-full cursor-pointer items-center justify-center" +
                  (checked ? " bg-white text-black" : "") +
                  (idx == 0 ? " rounded-l" : "") +
                  (idx == possibleValues.length - 1 ? " rounded-r" : "")
                }
              >
                {option}
              </label>
            )}
          </RadioGroup.Option>
        ))}
      </BorderBox>
    </RadioGroup>
  );
}
